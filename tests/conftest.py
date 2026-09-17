"""Shared test doubles: a tiny SteerViT stand-in so the node runs without weights or network.

The fake reproduces the interface the node relies on (``image_size``, ``patch_size``,
``vision_model.trunk.num_prefix_tokens``, ``get_transforms``, ``forward(images, texts,
return_segmentation_logits)``, ``get_heatmap_logits``) with deterministic, prompt-dependent
tokens, so golden tests can recompute every output by hand.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import cuvis_ai_steervit.node.steervit as steervit_mod

RES, PATCH, DIM = 28, 14, 8  # 2x2 patch grid, 8-dim tokens
FAKE_MEAN, FAKE_STD = (0.5, 0.4, 0.3), (0.2, 0.25, 0.3)


class Normalize:  # the node matches the class NAME of the model transform
    def __init__(self, mean, std) -> None:
        self.mean, self.std = mean, std


class _Transforms:
    def __init__(self) -> None:
        self.transforms = [object(), Normalize(FAKE_MEAN, FAKE_STD)]


class _Trunk:
    num_prefix_tokens = 1


class _Vision:
    def __init__(self) -> None:
        self.trunk = _Trunk()


class FakeSteerViT(nn.Module):
    """Deterministic stand-in: tokens = f(patch colour, prompt length); one linear head."""

    def __init__(self, seed: int = 0, with_transforms: bool = True) -> None:
        super().__init__()
        self.vision_model = _Vision()
        self.with_transforms = with_transforms
        g = torch.Generator().manual_seed(seed)
        self.head = nn.Linear(DIM, 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(1, DIM, generator=g))
            self.head.bias.fill_(0.1)
        self.calls: list[list[str]] = []

    @property
    def image_size(self) -> tuple[int, int]:
        return (RES, RES)

    @property
    def patch_size(self) -> int:
        return PATCH

    def get_transforms(self) -> _Transforms:
        if not self.with_transforms:
            raise RuntimeError("no transforms")
        return _Transforms()

    def forward(self, images, texts=None, return_segmentation_logits=False):
        self.calls.append(list(texts))
        b = images.shape[0]
        n = (RES // PATCH) ** 2
        pooled = F.avg_pool2d(images, PATCH).flatten(2).transpose(1, 2)  # [B, n, 3]
        offs = torch.tensor([float(len(t)) for t in texts], dtype=images.dtype).view(b, 1, 1) / 10.0
        ones = torch.ones(b, n, 1, dtype=images.dtype)
        tok = torch.cat([pooled, pooled * offs, offs.expand(b, n, 1), ones], dim=-1)  # [B, n, 8]
        out = torch.cat([torch.zeros(b, 1, DIM, dtype=images.dtype), tok], dim=1)  # prefix token
        return self.get_heatmap_logits(out) if return_segmentation_logits else out

    def get_heatmap_logits(self, img_feats):
        return self.head(img_feats[:, 1:, :]).squeeze(-1)


@pytest.fixture
def fake_loader(monkeypatch):
    """Route the node's model loader to the fake; returns the fake instances it created."""
    created: list[FakeSteerViT] = []

    def _load(checkpoint: str, hf_repo: str) -> FakeSteerViT:
        model = FakeSteerViT()
        created.append(model)
        return model

    monkeypatch.setattr(steervit_mod, "_load_steervit", _load)
    return created


def fake_preprocess(rgb_bhwc: torch.Tensor) -> torch.Tensor:
    """The node's preprocessing, spelled out independently for golden checks."""
    x = rgb_bhwc.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(FAKE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(FAKE_STD).view(1, 3, 1, 1)
    return (x - mean) / std
