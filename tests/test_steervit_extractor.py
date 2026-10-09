"""SteerViTExtractor with the fake backbone: golden outputs against the original text-conditioned
forward, prompt caching and batching, port contract, frozen state, serialization, validation."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import ExecutionStage
from torch import nn

import cuvis_ai_steervit.node.steervit as mod
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
    monkeypatch.setattr(mod, "_load_steervit", lambda c, r, v: FakeSteerViT(with_transforms=False))
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
        "hf_revision",
        "prompts",
        "feature_prompt",
        "topk_frac",
        "score_activation",
        "autocast_dtype",
        "tf32",
    ):
        assert key in hp, key
    json.dumps(hp)
    assert hp["prompts"] == ["p1", "p2"] and hp["feature_prompt"] == "p2"
    assert SteerViTExtractor(name="sv3").hparams["feature_prompt"] is None
    assert SteerViTExtractor(name="sv4").hparams["hf_revision"] == mod.DEFAULT_HF_REVISION


# ----- 4. validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"prompts": []},
        {"prompts": ["ok", "  "]},
        {"feature_prompt": ""},
        {"hf_revision": " "},
        {"topk_frac": 0.0},
        {"topk_frac": 1.5},
        {"score_activation": "softmax"},
        {"autocast_dtype": "float8"},
    ],
)
def test_invalid_hparams_raise(fake_loader, bad):
    with pytest.raises(ValueError):
        SteerViTExtractor(**bad)


# ----- 5. weights: the registry mirrors, other repositories, local files -------------------------


class _Recorder:
    """Stands in for ``_build_steervit``: records the checkpoint path it is asked to build from."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, path: str) -> FakeSteerViT:
        self.paths.append(path)
        return FakeSteerViT()


@pytest.fixture
def no_hub(monkeypatch):
    """Fail on any hub download."""
    import huggingface_hub

    def _no_download(**kwargs):
        raise AssertionError(f"unexpected hub download {kwargs}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_download)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """``ModelWeights.resolve`` stand-in: a file per registry name, and the names it was asked."""
    from cuvis_ai_core.data.model_weights import ModelWeights

    asked: list[str] = []

    def _resolve(name, **kwargs):
        asked.append(name)
        path = tmp_path / "cache" / name / "weights.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    monkeypatch.setattr(ModelWeights, "resolve", _resolve)
    return asked


def test_the_default_checkpoint_comes_from_the_registry(monkeypatch, tmp_path, registry, no_hub):
    loader = _Recorder()
    monkeypatch.setattr(mod, "_build_steervit", loader)
    SteerViTExtractor(name="sv")
    assert registry == ["steervit_dinov2_base"]
    assert loader.paths == [str(tmp_path / "cache" / "steervit_dinov2_base" / "weights.bin")]


def test_the_upstream_file_saved_pipelines_name_resolves_to_the_mirror(
    monkeypatch, registry, no_hub
):
    """Pipelines saved before the mirror record the upstream repository and its pinned commit."""
    loader = _Recorder()
    monkeypatch.setattr(mod, "_build_steervit", loader)
    SteerViTExtractor(hf_repo=mod.UPSTREAM_HF_REPO, hf_revision=mod.UPSTREAM_HF_REVISION, name="sv")
    assert registry == ["steervit_dinov2_base"]


def test_the_upstream_repository_without_a_revision_resolves_to_the_mirror(
    monkeypatch, registry, no_hub
):
    """Pipelines saved before ``hf_revision`` existed record ``hf_repo`` alone (every walnut
    catalog pipeline does); the revision then takes the node default."""
    loader = _Recorder()
    monkeypatch.setattr(mod, "_build_steervit", loader)
    node = SteerViTExtractor(hf_repo=mod.UPSTREAM_HF_REPO, name="sv")
    assert registry == ["steervit_dinov2_base"]
    assert node.hparams["hf_revision"] == mod.DEFAULT_HF_REVISION


def test_another_repository_or_revision_downloads_from_the_hub(monkeypatch, tmp_path, registry):
    import huggingface_hub

    calls = []
    local = tmp_path / "steervit_dinov2_base.pth"

    def _download(**kwargs):
        calls.append(kwargs)
        return str(local)

    loader = _Recorder()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(mod, "_build_steervit", loader)
    SteerViTExtractor(hf_repo="org/repo", hf_revision=None, name="sv")
    # the upstream repository at another revision is not the mirrored file
    SteerViTExtractor(hf_repo=mod.UPSTREAM_HF_REPO, hf_revision=None, name="sv2")
    assert calls == [
        {"repo_id": "org/repo", "filename": mod.DEFAULT_CHECKPOINT, "revision": None},
        {"repo_id": mod.UPSTREAM_HF_REPO, "filename": mod.DEFAULT_CHECKPOINT, "revision": None},
    ]
    assert registry == [] and loader.paths == [str(local)] * 2


def test_local_checkpoint_path_skips_the_registry_and_the_hub(
    monkeypatch, tmp_path, registry, no_hub
):
    local = tmp_path / "my_steervit.pth"
    local.write_bytes(b"")
    loader = _Recorder()
    monkeypatch.setattr(mod, "_build_steervit", loader)
    SteerViTExtractor(checkpoint=str(local), name="sv")
    assert loader.paths == [str(local)] and registry == []


class _Timm:
    """Records ``create_model`` calls in place of the vendored backbone's timm."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_model(self, model_name: str, **kwargs):
        self.calls.append((model_name, kwargs))
        return nn.Identity()


class _VendoredModel(nn.Module):
    """Stands in for the vendored ``SteerViT``: builds its trunk through the backbone's timm."""

    def __init__(self, config) -> None:
        super().__init__()
        from cuvis_ai_steervit._vendor.steervit import backbone

        self.config = config
        self.trunk = backbone.timm.create_model(
            config["vision_encoder"]["model_name"], pretrained=True, img_size=4
        )
        self.lin = nn.Linear(2, 1)


def _checkpoint(tmp_path, text_encoder: str, model_name: str) -> str:
    path = tmp_path / "ckpt.pth"
    torch.save(
        {
            "config": {
                "text_encoder": text_encoder,
                "vision_encoder": {"model_name": model_name},
            },
            "state_dict": {"lin.weight": torch.ones(1, 2), "lin.bias": torch.zeros(1)},
        },
        path,
    )
    return str(path)


@pytest.fixture
def vendored(monkeypatch):
    """The vendored SteerViT and the backbone's timm replaced by recorders."""
    import cuvis_ai_steervit._vendor.steervit as vendor
    from cuvis_ai_steervit._vendor.steervit import backbone

    timm = _Timm()
    monkeypatch.setattr(backbone, "timm", timm)
    monkeypatch.setattr(vendor, "SteerViT", _VendoredModel)
    return timm


def test_build_reads_the_mirrored_trunk_and_text_encoder(monkeypatch, tmp_path, vendored):
    from cuvis_ai_core.data.model_weights import ModelWeights

    from cuvis_ai_steervit._vendor.steervit import backbone

    snapshot = tmp_path / "models--cubert-gmbh--roberta-large" / "snapshots" / "rev"
    trunk = tmp_path / "trunk" / "model.safetensors"
    files = {
        "roberta_large": snapshot / "model.safetensors",
        "vit_base_patch14_dinov2_lvd142m": trunk,
    }
    monkeypatch.setattr(ModelWeights, "resolve", lambda name, **kwargs: files[name])
    model = mod._build_steervit(
        _checkpoint(tmp_path, "roberta-large", "vit_base_patch14_dinov2.lvd142m")
    )
    assert model.config["text_encoder"] == str(snapshot)
    assert vendored.calls == [
        (
            "vit_base_patch14_dinov2.lvd142m",
            {"pretrained": True, "img_size": 4, "pretrained_cfg_overlay": {"file": str(trunk)}},
        )
    ]
    assert backbone.timm is vendored  # the stand-in is gone again
    assert torch.equal(model.lin.weight, torch.ones(1, 2))
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_build_leaves_unmirrored_models_to_their_own_loaders(monkeypatch, tmp_path, vendored):
    from cuvis_ai_core.data.model_weights import ModelWeights

    def _unexpected(name, **kwargs):
        raise AssertionError(f"unexpected registry lookup {name}")

    monkeypatch.setattr(ModelWeights, "resolve", _unexpected)
    model = mod._build_steervit(
        _checkpoint(tmp_path, "roberta-base", "vit_small_patch14_dinov2.lvd142m")
    )
    assert model.config["text_encoder"] == "roberta-base"
    assert vendored.calls == [
        ("vit_small_patch14_dinov2.lvd142m", {"pretrained": True, "img_size": 4})
    ]


def test_the_trunk_stand_in_is_removed_after_an_error():
    from cuvis_ai_steervit._vendor.steervit import backbone

    real = backbone.timm
    with pytest.raises(RuntimeError, match="boom"):
        with mod._trunk_weights_from("trunk.safetensors"):
            assert backbone.timm is not real
            assert backbone.timm.__name__ == real.__name__  # everything else is the real timm
            raise RuntimeError("boom")
    assert backbone.timm is real


def test_text_encoder_folder_on_a_read_only_omegaconf_config():
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create({"text_encoder": "roberta-large"})
    omegaconf.OmegaConf.set_readonly(cfg, True)
    omegaconf.OmegaConf.set_struct(cfg, True)
    mod._set_config_value(cfg, "text_encoder", "/c/models--cubert-gmbh--roberta-large/snapshots/r")
    assert cfg["text_encoder"].endswith("snapshots/r")
    assert omegaconf.OmegaConf.is_readonly(cfg)


# ----- 6. reduced precision --------------------------------------------------------------------


def test_autocast_is_a_no_op_on_cpu_inputs(fake_loader):
    ref = SteerViTExtractor(prompts=("p1", "p2"), name="sv")
    amp = SteerViTExtractor(prompts=("p1", "p2"), autocast_dtype="float16", name="sv_amp")
    assert amp.hparams["autocast_dtype"] == "float16" and ref.hparams["autocast_dtype"] is None
    rgb = _rgb(3)
    a, b = ref(rgb_image=rgb), amp(rgb_image=rgb)
    for port in ("features", "scores", "anomaly_score"):
        assert torch.equal(a[port], b[port]), port


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autocast")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_autocast_on_cuda_keeps_float32_outputs_close_to_fp32(fake_loader, dtype):
    ref = SteerViTExtractor(prompts=("p1", "p2"), name="sv").cuda()
    amp = SteerViTExtractor(prompts=("p1", "p2"), autocast_dtype=dtype, name="sv_amp").cuda()
    rgb = _rgb(3).cuda()
    a, b = ref(rgb_image=rgb), amp(rgb_image=rgb)
    for port in ("features", "scores", "anomaly_score"):
        assert b[port].dtype == torch.float32, port
        assert torch.allclose(b[port], a[port], rtol=2e-2, atol=2e-2), port


def test_tf32_is_a_no_op_on_cpu_and_restores_the_process_setting(fake_loader):
    ref = SteerViTExtractor(prompts=("p1", "p2"), name="sv")
    tf = SteerViTExtractor(prompts=("p1", "p2"), tf32=True, name="sv_tf32")
    assert tf.hparams["tf32"] is True and ref.hparams["tf32"] is False
    before = torch.get_float32_matmul_precision()
    rgb = _rgb(4)
    a, b = ref(rgb_image=rgb), tf(rgb_image=rgb)
    assert torch.get_float32_matmul_precision() == before
    for port in ("features", "scores", "anomaly_score"):
        assert torch.equal(a[port], b[port]), port


def test_tf32_context_restores_on_error():
    before = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError), mod._tf32_matmul(True):
        assert torch.get_float32_matmul_precision() == "high"
        raise RuntimeError("boom")
    assert torch.get_float32_matmul_precision() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TF32")
def test_tf32_on_cuda_stays_close_to_fp32(fake_loader):
    ref = SteerViTExtractor(prompts=("p1", "p2"), name="sv").cuda()
    tf = SteerViTExtractor(prompts=("p1", "p2"), tf32=True, name="sv_tf32").cuda()
    rgb = _rgb(4).cuda()
    a, b = ref(rgb_image=rgb), tf(rgb_image=rgb)
    for port in ("features", "scores", "anomaly_score"):
        assert b[port].dtype == torch.float32 and torch.allclose(
            b[port], a[port], rtol=1e-2, atol=1e-2
        )
