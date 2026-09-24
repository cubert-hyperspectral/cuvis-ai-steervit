"""Reload smoke: stretch -> SteerViTExtractor (and the tiled multi-scale path stretch -> ImageTiler
-> SteerViTExtractor -> GridStitcher) inside a CuvisPipeline survives save_to_file ->
load_pipeline (yaml + .pt) and reproduces its outputs, with the model loader routed to the fake."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_steervit.node.steervit import SteerViTExtractor
from cuvis_ai_steervit.node.stretch import JointPercentileStretch
from cuvis_ai_steervit.node.tiling import GridStitcher, ImageTiler

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
H, W = 36, 44


class _ConstantRGBSource(Node):
    """Module-scope test source: a deterministic random 3-band frame in reflectance units."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"rgb": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 3))}

    def __init__(self, seed: int = 0, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.seed = int(seed)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed)
        return {"rgb": torch.rand(1, H, W, 3, generator=g) * 4000.0 + 200.0}


def test_stretch_and_extractor_pipeline_reloads(tmp_path, fake_loader):
    src = _ConstantRGBSource(seed=3, name="src")
    stretch = JointPercentileStretch(quantize_levels=255, name="stretch")
    sv = SteerViTExtractor(prompts=["a", "bb"], feature_prompt="bb", name="sv")
    pipe = CuvisPipeline("steervit_reload_smoke")
    pipe.connect(src.outputs.rgb, stretch.inputs.data)
    pipe.connect(stretch.outputs.normalized, sv.inputs.rgb_image)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    before = pipe.forward(batch={}, context=ctx)
    assert before[("sv", "features")].shape == (1, 2, 2, 8)
    assert before[("sv", "scores")].shape == (1, H, W, 1)

    yaml_path = tmp_path / "sv.yaml"
    pipe.save_to_file(str(yaml_path))
    pt_path = yaml_path.with_suffix(".pt")
    assert pt_path.exists(), "save_to_file must write the .pt next to the yaml"
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["steervit"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path), weights_path=str(pt_path), device="cpu", node_registry=registry
    )
    restored_sv = next(n for n in restored.nodes if not isinstance(n, str) and n.name == "sv")
    assert isinstance(restored_sv, SteerViTExtractor)
    assert restored_sv.hparams["prompts"] == ["a", "bb"]
    assert restored_sv.hparams["feature_prompt"] == "bb"
    after = restored.forward(batch={}, context=ctx)
    for key in ("features", "scores", "anomaly_score"):
        assert torch.allclose(after[("sv", key)], before[("sv", key)], atol=1e-6), key


def test_tiled_extractor_pipeline_reloads(tmp_path, fake_loader):
    src = _ConstantRGBSource(seed=5, name="src")
    stretch = JointPercentileStretch(quantize_levels=255, name="stretch")
    tiler = ImageTiler(tiles=2, name="tiler")
    sv = SteerViTExtractor(prompts=["a"], name="sv")
    stitch = GridStitcher(tiles=2, name="stitch")
    pipe = CuvisPipeline("steervit_tiled_reload_smoke")
    pipe.connect(src.outputs.rgb, stretch.inputs.data)
    pipe.connect(stretch.outputs.normalized, tiler.inputs.image)
    pipe.connect(tiler.outputs.tiles, sv.inputs.rgb_image)
    pipe.connect(sv.outputs.features, stitch.inputs.tiles)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    before = pipe.forward(batch={}, context=ctx)
    assert before[("tiler", "tiles")].shape == (4, H // 2, W // 2, 3)
    assert before[("stitch", "grid")].shape == (1, 4, 4, 8)  # 2 x 2 tiles of 2 x 2 patches

    yaml_path = tmp_path / "tiled.yaml"
    pipe.save_to_file(str(yaml_path))
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["steervit"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path),
        weights_path=str(yaml_path.with_suffix(".pt")),
        device="cpu",
        node_registry=registry,
    )
    nodes = {n.name: n for n in restored.nodes if not isinstance(n, str)}
    assert isinstance(nodes["tiler"], ImageTiler) and nodes["tiler"].tiles == 2
    assert isinstance(nodes["stitch"], GridStitcher) and nodes["stitch"].tiles == 2
    after = restored.forward(batch={}, context=ctx)
    assert torch.equal(after[("tiler", "tiles")], before[("tiler", "tiles")])
    assert torch.allclose(after[("stitch", "grid")], before[("stitch", "grid")], atol=1e-6)
