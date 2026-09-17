"""SteerViTExtractor with the fake backbone: golden outputs against the original text-conditioned
forward, prompt caching and batching, port contract, frozen state, serialization, validation."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_steervit.node.steervit import SteerViTExtractor
from tests.conftest import DIM, FakeSteerViT, fake_preprocess

pytestmark = pytest.mark.unit

B, H, W = 2, 40, 30
PROMPTS = ["a", "bb", "ccc"]
G = 2  # RES // PATCH


def _rgb(seed: int = 0) -> torch.Tensor:
    return torch.rand(B, H, W, 3, generator=torch.Generator().manual_seed(seed))


def _reference(model: FakeSteerViT, rgb: torch.Tensor, prompts: list[str], feature_prompt: str):
    """The ORIGINAL SteerViT path: tokenise + text tower + vision per forward call, no caching."""
    x = fake_preprocess(rgb)
    b = rgb.shape[0]
    tok = model.forward(x.repeat_interleave(len(prompts), 0), texts=prompts * b)
    logits = model.get_heatmap_logits(tok).reshape(b, len(prompts), G * G)
    grid = torch.sigmoid(logits).mean(1).reshape(b, 1, G, G)
    scores = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
    ftok = model.forward(x, texts=[feature_prompt] * b)  # its own tokenisation / padding
    feats = ftok[:, 1:, :].reshape(b, G, G, DIM)
    return scores.permute(0, 2, 3, 1), feats


# ----- 1. golden: cached prompt encodings reproduce the original forward --------------------------


def test_golden_scores_and_features(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, name="sv")
    rgb = _rgb()
    out = node(rgb_image=rgb)
    scores, feats = _reference(FakeSteerViT(), rgb, PROMPTS, PROMPTS[0])
    assert torch.allclose(out["scores"], scores, atol=1e-6)
    assert torch.allclose(out["features"], feats, atol=1e-6)
    k = max(1, int(0.001 * H * W))
    topk = torch.topk(out["scores"].reshape(B, -1), k, dim=1).values.mean(1)
    assert torch.allclose(out["anomaly_score"], topk)
    # one batched vision call covers the whole ensemble: B images x 3 prompts, padded to "ccc"
    assert fake_loader[0].vision_model.calls == [(B * 3, 3)]


def test_feature_prompt_inside_ensemble_reuses_the_pass(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, feature_prompt="ccc", name="sv")
    rgb = _rgb(1)
    out = node(rgb_image=rgb)
    assert len(fake_loader[0].vision_model.calls) == 1
    # padding-invariance: "ccc" encoded inside the 3-prompt batch equals "ccc" encoded alone
    _, feats = _reference(FakeSteerViT(), rgb, PROMPTS, "ccc")
    assert torch.allclose(out["features"], feats, atol=1e-6)


def test_feature_prompt_outside_ensemble_costs_one_extra_pass(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, feature_prompt="zzzz", name="sv")
    rgb = _rgb(2)
    out = node(rgb_image=rgb)
    calls = fake_loader[0].vision_model.calls
    assert len(calls) == 2 and calls[1] == (B, 4)  # B images, the 4-token extra prompt
    _, feats = _reference(FakeSteerViT(), rgb, PROMPTS, "zzzz")
    assert torch.allclose(out["features"], feats, atol=1e-6)
    assert "_feature_feats" in node.state_dict() and "_feature_mask" in node.state_dict()


def test_score_activation_none_averages_raw_logits(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, score_activation="none", name="sv")
    rgb = _rgb(3)
    out = node(rgb_image=rgb)
    model = FakeSteerViT()
    tok = model.forward(fake_preprocess(rgb).repeat_interleave(3, 0), texts=PROMPTS * B)
    grid = model.get_heatmap_logits(tok).reshape(B, 3, G * G).mean(1).reshape(B, 1, G, G)
    expected = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
    assert torch.allclose(out["scores"], expected.permute(0, 2, 3, 1), atol=1e-6)


def test_normalization_falls_back_to_imagenet_without_transforms(fake_loader, monkeypatch):
    import cuvis_ai_steervit.node.steervit as mod

    monkeypatch.setattr(mod, "_load_steervit", lambda c, r: FakeSteerViT(with_transforms=False))
    node = SteerViTExtractor(name="sv")
    assert torch.allclose(node._mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    assert torch.allclose(node._std.flatten(), torch.tensor([0.229, 0.224, 0.225]))


# ----- 2. port contract -------------------------------------------------------------------------


def test_port_contract_and_metadata(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, name="sv")
    out = node(rgb_image=_rgb())
    assert set(out) == set(node.OUTPUT_SPECS)
    assert out["features"].shape == (B, G, G, DIM)
    assert out["scores"].shape == (B, H, W, 1)
    assert out["anomaly_score"].shape == (B,)
    for k in out:
        assert out[k].dtype == node.OUTPUT_SPECS[k].dtype
        assert out[k].grad_fn is None  # inference-only path
    assert node.grid_size == G and node.INPUT_SPECS["rgb_image"].shape[-1] == 3
    assert node.requires_initial_fit is False and node.TRAINABLE_BUFFERS == ()
    assert ExecutionStage.ALWAYS in node.execution_stages


def test_batch_matches_per_sample(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, name="sv")
    rgb = _rgb(4)
    batched = node(rgb_image=rgb)
    for i in range(B):
        single = node(rgb_image=rgb[i : i + 1])
        assert torch.allclose(batched["scores"][i], single["scores"][0], atol=1e-6)
        assert torch.allclose(batched["features"][i], single["features"][0], atol=1e-6)


# ----- 3. frozen state, cached prompts, serialization -------------------------------------------


def test_frozen_model_stays_in_eval_mode_under_train(fake_loader):
    node = SteerViTExtractor(name="sv")
    assert not node._model.training
    node.train()
    assert node.training and not node._model.training  # the frozen tower never flips
    node.eval()
    assert not node._model.training


def test_text_tower_is_dropped_and_prompts_are_cached_buffers(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS, name="sv")
    assert not hasattr(node._model, "text_model") and node._model.tokenizer is None
    assert node._prompt_feats.shape == (3, 3, DIM)  # [P, L (padded to "ccc"), D]
    assert node._prompt_mask.shape == (3, node._model.num_img_tokens + 3)
    assert (
        node._prompt_mask.dtype == torch.bool
        and node._prompt_mask[:, : node._model.num_img_tokens].all()
    )
    keys = set(node.state_dict())
    assert {"_prompt_feats", "_prompt_mask"} <= keys
    assert not any("text_model" in k for k in keys)


def test_weights_are_frozen_and_travel_in_the_state_dict(fake_loader):
    node = SteerViTExtractor(name="sv")
    assert all(not p.requires_grad for p in node.parameters())  # frozen by the node, not the loader
    keys = set(node.state_dict())
    assert {"_model.head.weight", "_model.head.bias", "_model.connector.weight"} <= keys
    assert not any(
        k.endswith(("_mean", "_std")) for k in keys
    )  # preprocessing constants, not state
    fresh = SteerViTExtractor(name="sv2")
    fresh.load_state_dict(node.state_dict())
    rgb = _rgb(5)
    assert torch.equal(fresh(rgb_image=rgb)["scores"], node(rgb_image=rgb)["scores"])


def test_hparams_are_json_serializable_and_complete(fake_loader):
    node = SteerViTExtractor(
        checkpoint="x.pth", hf_repo="org/repo", prompts=("p1", "p2"), feature_prompt="p2", name="sv"
    )
    hp = node.hparams
    for key in (
        "checkpoint",
        "hf_repo",
        "prompts",
        "feature_prompt",
        "topk_frac",
        "score_activation",
    ):
        assert key in hp, key
    json.dumps(hp)
    assert hp["prompts"] == ["p1", "p2"] and hp["feature_prompt"] == "p2"
    assert SteerViTExtractor(name="sv3").hparams["feature_prompt"] is None


# ----- 4. validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"prompts": []},
        {"prompts": ["ok", "  "]},
        {"feature_prompt": ""},
        {"topk_frac": 0.0},
        {"topk_frac": 1.5},
        {"score_activation": "softmax"},
    ],
)
def test_invalid_hparams_raise(fake_loader, bad):
    with pytest.raises(ValueError):
        SteerViTExtractor(**bad)
