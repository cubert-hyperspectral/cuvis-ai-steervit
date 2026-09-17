"""Real-weights parity (slow): the node, built from the published checkpoint in THIS environment,
reproduces the golden reference computed with the same weights in the validated reference
environment (steervit 0.2.0 install, transformers 4.x, CPU). Guards the preprocessing, the
prompt batching and the transformers-major compatibility at once.

Run with ``pytest -m slow``; needs the Hugging Face hub (or a warm cache) for the checkpoint, the
timm DINOv2 trunk and roberta-base.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cuvis_ai_steervit.node.steervit import SteerViTExtractor

pytestmark = pytest.mark.slow

GOLDEN = Path(__file__).resolve().parent / "data" / "steervit_golden.pt"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():
        pytest.skip("golden reference not present")
    return torch.load(GOLDEN, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def node(golden) -> SteerViTExtractor:
    try:
        return SteerViTExtractor(prompts=golden["prompts"], name="sv").eval()
    except Exception as exc:  # noqa: BLE001 - offline machines skip rather than fail
        pytest.skip(f"SteerViT weights unavailable: {exc}")


def test_geometry_matches_reference(node, golden):
    assert node._resolution == golden["resolution"]
    assert node.grid_size == golden["grid"]
    assert node._num_prefix == golden["num_prefix_tokens"]
    assert torch.allclose(node._mean.flatten(), torch.tensor(golden["mean"]))
    assert torch.allclose(node._std.flatten(), torch.tensor(golden["std"]))


def test_prompted_map_and_features_match_reference(node, golden):
    img = golden["image"]
    with torch.no_grad():
        patch, logits = node._run(node._preprocess(img), node.prompts)
    grid = torch.sigmoid(logits).reshape(1, len(node.prompts), node.grid_size, node.grid_size)
    assert torch.allclose(grid[0], golden["scores_grid_per_prompt"], atol=2e-3), (
        (grid[0] - golden["scores_grid_per_prompt"]).abs().max()
    )
    f0 = patch[0, 0]  # prompt-0 tokens [G*G, D]
    assert torch.allclose(f0[:16], golden["features_prompt0_first16"], atol=5e-3, rtol=1e-3)
    assert torch.allclose(f0.norm(dim=1), golden["features_prompt0_token_norms"], rtol=1e-3)
    assert torch.allclose(f0.mean(0), golden["features_prompt0_mean"], atol=5e-3)

    out = node(rgb_image=img)
    assert out["features"].shape == (1, node.grid_size, node.grid_size, f0.shape[1])
    assert torch.allclose(out["features"][0].reshape(-1, f0.shape[1]), f0, atol=1e-5)
    assert out["scores"].shape == (1, img.shape[1], img.shape[2], 1)
