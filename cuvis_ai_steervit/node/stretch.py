"""Per-frame percentile stretch to ``[0, 1]``, jointly over all channels.

Pretrained RGB backbones expect a natural-looking image; a reflectance projection has an arbitrary
scale and a few specular highlights. Stretching each frame between its low and high percentiles,
computed over ALL channels together, keeps the colour ratios between channels (a per-channel
stretch would re-balance them) and clips the highlights. It is the preprocessing the validated
SteerViT feature bank was fitted with, and it is stateless: nothing is fitted, nothing drifts
between sessions.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


def _percentile(flat: Tensor, q: float) -> Tensor:
    """Linear-interpolation percentile of a 1-D tensor (numpy's default method), any length."""
    n = flat.numel()
    s, _ = torch.sort(flat)
    pos = torch.tensor(q / 100.0 * (n - 1), dtype=s.dtype, device=s.device)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo.to(s.dtype))


class JointPercentileStretch(Node):
    """Stretch each frame of a BHWC tensor to [0, 1] between two joint-channel percentiles."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.PREPROCESSING, NodeTag.NORMALIZATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Any BHWC tensor, e.g. a 3-band false-RGB projection in reflectance units.",
        ),
    }
    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Same shape, per frame (x - p_low) / (p_high - p_low) clipped to [0, 1]; "
            "percentiles over all channels jointly (or per channel), optionally quantized.",
        ),
    }

    def __init__(
        self,
        low: float = 2.0,
        high: float = 98.0,
        per_channel: bool = False,
        quantize_levels: int | None = None,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Create a stretch node.

        Parameters
        ----------
        low, high : percentiles in ``[0, 100]`` mapped to 0 and 1 (``2`` / ``98`` = the validated
            false-RGB preprocessing).
        per_channel : take the percentiles per channel instead of jointly over all channels.
        quantize_levels : if set (e.g. ``255``), truncate the result to that many levels, exactly
            as casting a ``[0, 1]`` image to ``uint8`` does; ``None`` keeps the float values.
        eps : floor for the ``(p_high - p_low)`` denominator.
        """
        if not 0.0 <= float(low) < float(high) <= 100.0:
            raise ValueError(
                f"JointPercentileStretch: require 0 <= low < high <= 100, got {low}, {high}"
            )
        if quantize_levels is not None and (
            isinstance(quantize_levels, bool) or int(quantize_levels) < 1
        ):
            raise ValueError(
                "JointPercentileStretch: quantize_levels must be a positive integer or None"
            )
        self.low = float(low)
        self.high = float(high)
        self.per_channel = bool(per_channel)
        self.quantize_levels = int(quantize_levels) if quantize_levels is not None else None
        self.eps = float(eps)
        super().__init__(
            low=self.low,
            high=self.high,
            per_channel=self.per_channel,
            quantize_levels=self.quantize_levels,
            eps=self.eps,
            **kwargs,
        )

    def _stretch(self, frame: Tensor) -> Tensor:
        """Stretch one [H, W, C] frame."""
        if self.per_channel:
            cols = frame.reshape(-1, frame.shape[-1])
            lo = torch.stack([_percentile(cols[:, c], self.low) for c in range(cols.shape[1])])
            hi = torch.stack([_percentile(cols[:, c], self.high) for c in range(cols.shape[1])])
        else:
            flat = frame.reshape(-1)
            lo, hi = _percentile(flat, self.low), _percentile(flat, self.high)
        out = ((frame - lo) / (hi - lo).clamp_min(self.eps)).clamp(0.0, 1.0)
        if self.quantize_levels is not None:  # truncation, as a uint8 cast of a [0, 1] image does
            out = torch.floor(out * self.quantize_levels) / self.quantize_levels
        return out

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Stretch every frame of the batch independently."""
        return {"normalized": torch.stack([self._stretch(frame) for frame in data], dim=0)}
