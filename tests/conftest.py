"""Shared test doubles: a tiny SteerViT stand-in so the node runs without weights or network.

The fake reproduces the interface the node relies on: ``image_size``, ``patch_size``,
``num_img_tokens``, ``vision_model(images, text_feats, attn_mask)`` with
``vision_model.trunk.num_prefix_tokens``, ``tokenizer`` / ``text_model`` / ``connector`` (the text
side that the node caches at construction), ``get_transforms`` and ``get_heatmap_logits``. Tokens
are deterministic functions of the patch colour and the prompt encoding, so golden tests can
recompute every output by hand.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import cuvis_ai_steervit.node.steervit as steervit_mod

RES, PATCH, DIM, TDIM = 28, 14, 8, 4  # 2x2 patch grid, 8-dim tokens, 4-dim text features
FAKE_MEAN, FAKE_STD = (0.5, 0.4, 0.3), (0.2, 0.25, 0.3)


class Normalize:  # the node matches the class NAME of the model transform
    def __init__(self, mean, std) -> None:
        self.mean, self.std = mean, std


class _Transforms:
    def __init__(self) -> None:
        self.transforms = [object(), Normalize(FAKE_MEAN, FAKE_STD)]


class _Tokenizer:
    """Character codes as token ids, padded to the longest prompt (like a real tokenizer)."""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        ids = [[ord(c) % 97 + 1 for c in t][:max_length] for t in texts]
        length = max(len(i) for i in ids)
        input_ids = torch.tensor([i + [0] * (length - len(i)) for i in ids])
        return {"input_ids": input_ids, "attention_mask": (input_ids > 0).long()}


class _TextModel(nn.Module):
    """Deterministic 'embedding': hidden state = id / 100 broadcast over TDIM dims."""

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, TDIM) / 100.0
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class _Trunk:
    num_prefix_tokens = 1


class _Vision(nn.Module):
    """Tokens = f(patch colour, prompt): the prompt enters through its first connector feature."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = _Trunk()
        self.calls: list[tuple[int, int]] = []

    def forward(self, images, text_feats, attn_mask=None):
        self.calls.append((int(images.shape[0]), int(text_feats.shape[1])))
        b = images.shape[0]
        n = (RES // PATCH) ** 2
        pooled = F.avg_pool2d(images, PATCH).flatten(2).transpose(1, 2)  # [B, n, 3]
        offs = text_feats[:, 0, 0].reshape(b, 1, 1).to(images.dtype)  # prompt-dependent scalar
        ones = torch.ones(b, n, 1, dtype=images.dtype)
        tok = torch.cat([pooled, pooled * offs, offs.expand(b, n, 1), ones], dim=-1)  # [B, n, 8]
        return torch.cat([torch.zeros(b, 1, DIM, dtype=images.dtype), tok], dim=1)  # + prefix


class FakeSteerViT(nn.Module):
    """Deterministic stand-in with the text side the node caches and the vision side it runs."""

    def __init__(self, seed: int = 0, with_transforms: bool = True) -> None:
        super().__init__()
        self.vision_model = _Vision()
        self.tokenizer = _Tokenizer()
        self.text_model = _TextModel()
        self.with_transforms = with_transforms
        g = torch.Generator().manual_seed(seed)
        self.connector = nn.Linear(TDIM, DIM)
        self.head = nn.Linear(DIM, 1)
        with torch.no_grad():
            self.connector.weight.copy_(torch.randn(DIM, TDIM, generator=g))
            self.connector.bias.zero_()
            self.head.weight.copy_(torch.randn(1, DIM, generator=g))
            self.head.bias.fill_(0.1)

    @property
    def image_size(self) -> tuple[int, int]:
        return (RES, RES)

    @property
    def patch_size(self) -> int:
        return PATCH

    @property
    def num_img_tokens(self) -> int:
        return (RES // PATCH) ** 2 + 1

    def get_transforms(self) -> _Transforms:
        if not self.with_transforms:
            raise RuntimeError("no transforms")
        return _Transforms()

    def forward(self, images, texts=None, return_segmentation_logits=False):
        """The original text-conditioned forward (reference for the cached path)."""
        feats, mask = steervit_mod._encode_prompts(self, list(texts))
        out = self.vision_model(images, feats, attn_mask=mask)
        return self.get_heatmap_logits(out) if return_segmentation_logits else out

    def get_heatmap_logits(self, img_feats):
        return self.head(img_feats[:, 1:, :]).squeeze(-1)


@pytest.fixture
def fake_loader(monkeypatch):
    """Route the node's model loader to the fake; returns the fake instances it created."""
    created: list[FakeSteerViT] = []

    def _load(checkpoint: str, hf_repo: str, hf_revision: str | None) -> FakeSteerViT:
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
