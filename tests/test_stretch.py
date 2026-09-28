"""JointPercentileStretch: golden parity with the numpy formula of the validated false-RGB path,
per-channel variant, 8-bit truncation, degenerate frames, port contract, hparam validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cuvis_ai_steervit.node.stretch import JointPercentileStretch

pytestmark = pytest.mark.unit

B, H, W, C = 2, 17, 13, 3


def _data(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 3.0, 0.5])  # anisotropic channels: joint != per-channel
    return torch.rand(B, H, W, C, generator=g) * scale * 4000.0 + 300.0


def _numpy_joint(frame: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo, hi = np.percentile(frame, lo_q), np.percentile(frame, hi_q)
    return np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)


def test_joint_stretch_matches_numpy_percentiles():
    data = _data()
    out = JointPercentileStretch(low=2.0, high=98.0)(data=data)["normalized"]
    for b in range(B):
        expected = _numpy_joint(data[b].numpy().astype(np.float64), 2.0, 98.0)
        assert np.allclose(out[b].numpy(), expected, atol=1e-5)


def test_per_channel_matches_numpy_per_channel():
    data = _data(1)
    out = JointPercentileStretch(low=1.0, high=99.0, per_channel=True)(data=data)["normalized"]
    for b in range(B):
        frame = data[b].numpy().astype(np.float64)
        lo = np.percentile(frame.reshape(-1, C), 1.0, axis=0)
        hi = np.percentile(frame.reshape(-1, C), 99.0, axis=0)
        expected = np.clip((frame - lo) / (hi - lo), 0, 1)
        assert np.allclose(out[b].numpy(), expected, atol=1e-5)
    joint = JointPercentileStretch(low=1.0, high=99.0)(data=data)["normalized"]
    assert not torch.allclose(joint, out)


def test_quantize_255_reproduces_the_uint8_image_path():
    data = _data(2)
    out = JointPercentileStretch(quantize_levels=255)(data=data)["normalized"]
    for b in range(B):
        stretched = _numpy_joint(data[b].numpy().astype(np.float64), 2.0, 98.0)
        as_png = (stretched * 255).astype(np.uint8).astype(np.float64) / 255.0  # truncation
        assert np.allclose(out[b].numpy(), as_png, atol=1e-6)


def test_frames_are_independent_and_constant_frame_is_finite():
    data = _data(3)
    node = JointPercentileStretch()
    batched = node(data=data)["normalized"]
    for b in range(B):
        assert torch.equal(batched[b], node(data=data[b : b + 1])["normalized"][0])
    flat = node(data=torch.full((1, H, W, C), 7.0))["normalized"]
    assert torch.isfinite(flat).all() and torch.equal(flat, torch.zeros_like(flat))


def test_port_contract():
    node = JointPercentileStretch()
    out = node(data=_data())
    assert set(out) == {"normalized"}
    assert out["normalized"].shape == (B, H, W, C)
    assert out["normalized"].dtype == node.OUTPUT_SPECS["normalized"].dtype
    assert 0.0 <= out["normalized"].min() and out["normalized"].max() <= 1.0
    assert node.requires_initial_fit is False and node.TRAINABLE_BUFFERS == ()


@pytest.mark.parametrize(
    "bad",
    [{"low": -1.0}, {"low": 50.0, "high": 50.0}, {"high": 101.0}, {"quantize_levels": 0}],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        JointPercentileStretch(**bad)


def test_hparams_json_round_trip():
    node = JointPercentileStretch(low=2, high=98, quantize_levels=255, name="stretch")
    hp = node.hparams
    json.dumps(hp)
    assert hp["low"] == 2.0 and hp["quantize_levels"] == 255 and hp["per_channel"] is False


def _sort_percentile(flat: torch.Tensor, q: float) -> torch.Tensor:
    """The former implementation (one sort per percentile), the reference for the one-sort one."""
    n = flat.numel()
    s, _ = torch.sort(flat)
    pos = torch.tensor(q / 100.0 * (n - 1), dtype=s.dtype, device=s.device)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo.to(s.dtype))


@pytest.mark.parametrize("n", [1, 2, 7, 1000, 1000 * 1080 * 3])
@pytest.mark.parametrize("q", [0.0, 2.0, 37.5, 50.0, 98.0, 100.0])
def test_one_sort_percentiles_are_bit_identical_to_one_sort_per_percentile(n, q):
    from cuvis_ai_steervit.node.stretch import _percentile, _percentiles

    g = torch.Generator().manual_seed(n)
    flat = torch.rand(n, generator=g) * 4000.0
    if n > 2:
        flat[: n // 3] = flat[0]  # ties, as in a clipped or quantised frame
    assert torch.equal(_percentile(flat, q), _sort_percentile(flat, q))
    lo, hi = _percentiles(flat, (q, 100.0 - q))
    assert torch.equal(lo, _sort_percentile(flat, q))
    assert torch.equal(hi, _sort_percentile(flat, 100.0 - q))
