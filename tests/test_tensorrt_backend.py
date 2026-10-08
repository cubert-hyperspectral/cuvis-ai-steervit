"""TensorRT backend of SteerViTExtractor and the engine helpers, with the fake backbone.

Mocked tests (no TensorRT, no GPU) check the plumbing: the exported pass computes exactly the node's
model call, so an engine that returns its outputs gives the torch backend's outputs bit for bit; one
engine per batch size; the precision follows the node's options; engines are keyed by a fingerprint
of the weights and prompts; a missing engine names the build command; new weights drop the loaded
engines; the export leaves the node's model as it was; the build writes the engine plus its record;
the CLI finds each node's batch from the pipeline graph. The ``slow`` test builds and runs real
engines of the fake backbone where CUDA, TensorRT and onnx are available.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest
import torch
import yaml

from cuvis_ai_steervit import trt_engine
from cuvis_ai_steervit.node.steervit import SteerViTExtractor
from tests.conftest import RES

pytestmark = pytest.mark.unit
CUDA = torch.cuda.is_available()
PROMPTS = ["a", "bb"]


def _rgb(b: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(b, 40, 30, 3, generator=torch.Generator().manual_seed(seed))


class FakeEngine:
    """Stands in for a TensorRT engine: runs the exported pass in torch, records its inputs."""

    device = torch.device("cpu")

    def __init__(self, graph):
        self.graph = graph
        self.inputs: list[torch.Tensor] = []

    def __call__(self, x):
        self.inputs.append(x)
        with torch.no_grad():
            tokens, logits = self.graph(x.contiguous())
        return {"tokens": tokens, "logits": logits}


def _trt_node(**kw) -> SteerViTExtractor:
    return SteerViTExtractor(prompts=PROMPTS, backend="tensorrt", name="steer", **kw)


# ------------------------------------------------------------------ options
def test_backend_defaults_hparams_and_validation(fake_loader):
    node = SteerViTExtractor(prompts=PROMPTS)
    assert node.backend == "torch" and node.engine_dir is None
    assert node.hparams["backend"] == "torch" and node.hparams["engine_dir"] is None
    trt = _trt_node(engine_dir="/engines", feature_prompt="bb")
    assert trt.hparams["backend"] == "tensorrt" and trt.hparams["engine_dir"] == "/engines"
    json.dumps(trt.hparams)
    with pytest.raises(ValueError, match="backend"):
        SteerViTExtractor(prompts=PROMPTS, backend="onnx")
    with pytest.raises(ValueError, match="bfloat16"):
        _trt_node(autocast_dtype="bfloat16")
    with pytest.raises(ValueError, match="feature_prompt"):
        _trt_node(feature_prompt="not a prompt")
    with pytest.raises(ValueError, match="engine_dir"):
        SteerViTExtractor(prompts=PROMPTS, engine_dir="")


@pytest.mark.parametrize(
    "kw, precision",
    [
        ({}, "fp32"),
        ({"tf32": True}, "tf32"),
        ({"autocast_dtype": "float16"}, "fp16"),
        ({"autocast_dtype": "fp16", "tf32": True}, "fp16"),
    ],
)
def test_engine_precision_follows_the_node_options(fake_loader, kw, precision):
    assert _trt_node(**kw).engine_precision == precision


# ------------------------------------------------------------------ the exported pass
@pytest.mark.parametrize("b", [1, 3])
def test_steered_pass_is_the_node_model_call(fake_loader, b):
    node = SteerViTExtractor(prompts=PROMPTS)
    x = node._preprocess(_rgb(b))
    want_patch, want_logits = node._run(x, node._prompt_feats, node._prompt_mask)
    tokens, logits = node._steered_pass(b)(x)
    assert torch.equal(tokens, want_patch) and torch.equal(logits, want_logits)


@pytest.mark.parametrize("b", [1, 4])
def test_tensorrt_forward_equals_torch_forward(fake_loader, monkeypatch, b):
    ref = SteerViTExtractor(prompts=PROMPTS, feature_prompt="bb")
    node = _trt_node(feature_prompt="bb")
    engines = {}

    def load(batch, device):
        engines[batch] = FakeEngine(node._steered_pass(batch))
        return engines[batch]

    monkeypatch.setattr(node, "_load_engine", load)
    rgb = _rgb(b, seed=b)
    got, want = node(rgb_image=rgb), ref(rgb_image=rgb)
    for port in ("features", "scores", "anomaly_score"):
        assert torch.equal(got[port], want[port]), port
    assert tuple(engines[b].inputs[0].shape) == (b, 3, RES, RES)
    node(rgb_image=_rgb(1)) if b != 1 else node(rgb_image=_rgb(2, seed=9))
    assert set(node._engines) == ({b, 1} if b != 1 else {1, 2})  # one engine per batch size


def test_engine_outputs_are_copied_out_of_the_reused_buffers(fake_loader, monkeypatch):
    node = _trt_node()
    buf = {}

    class ReusingEngine(FakeEngine):
        def __call__(self, x):
            out = super().__call__(x)
            for k, v in out.items():  # like TensorRT: always the same output tensors
                buf.setdefault(k, torch.empty_like(v)).copy_(v)
            return buf

    monkeypatch.setattr(
        node, "_load_engine", lambda batch, device: ReusingEngine(node._steered_pass(batch))
    )
    first = node(rgb_image=_rgb(1, seed=1))["features"].clone()
    held = node(rgb_image=_rgb(1, seed=1))["features"]
    node(rgb_image=_rgb(1, seed=5))
    assert torch.equal(held, first)  # an earlier output is not overwritten by a later call


def test_new_weights_drop_the_loaded_engines(fake_loader):
    node = _trt_node()
    node._engines = {1: object()}
    node._fingerprint = "old"
    node.load_state_dict(node.state_dict())
    assert node._engines == {} and node._fingerprint is None


# ------------------------------------------------------------------ fingerprints, names, loading
@pytest.fixture
def fake_trt(monkeypatch):
    """A minimal ``tensorrt`` module: records builder flags, returns a fake serialized engine."""

    class BuilderFlag:
        TF32, FP16 = "TF32", "FP16"

    class Logger:
        WARNING = 1

        def __init__(self, level):
            self.level = level

    class Config:
        def __init__(self):
            self.flags = {BuilderFlag.TF32}

        def set_flag(self, flag):
            self.flags.add(flag)

        def clear_flag(self, flag):
            self.flags.discard(flag)

    class Builder:
        blob: bytes | None = b"engine"
        configs: list[Config] = []

        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags):
            return object()

        def create_builder_config(self):
            Builder.configs.append(Config())
            return Builder.configs[-1]

        def build_serialized_network(self, network, config):
            return Builder.blob

    class OnnxParser:
        ok = True

        def __init__(self, network, logger):
            self.num_errors = 0 if OnnxParser.ok else 1

        def parse_from_file(self, path):
            assert os.path.exists(path)
            return OnnxParser.ok

        def get_error(self, i):
            return "bad node"

    trt = types.SimpleNamespace(
        __version__="10.15.1.29",
        BuilderFlag=BuilderFlag,
        Logger=Logger,
        Builder=Builder,
        OnnxParser=OnnxParser,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(trt_engine, "gpu_tag", lambda device=None: "Fake-GPU-sm00")
    monkeypatch.setattr(trt_engine, "_LOGGER", None)
    return trt


def test_engine_file_name_and_default_dir(fake_trt, monkeypatch, tmp_path):
    name = trt_engine.engine_file_name("fp16", "0123456789abcdef", 4, 336)
    assert name == "fp16_0123456789abcdef_b4_r336_Fake-GPU-sm00_trt10.15.1.29.engine"
    monkeypatch.setenv(trt_engine.ENGINE_DIR_ENV, str(tmp_path))
    assert trt_engine.default_engine_dir() == os.path.join(str(tmp_path), "steervit")
    monkeypatch.delenv(trt_engine.ENGINE_DIR_ENV)
    assert trt_engine.default_engine_dir().endswith(
        os.path.join(".cache", "cuvis-ai", "tensorrt", "steervit")
    )


def test_engine_path_tracks_batch_weights_and_prompts(fake_loader, fake_trt, tmp_path):
    node = _trt_node(engine_dir=str(tmp_path))
    p1, p4 = node.engine_path(1), node.engine_path(4)
    assert os.path.dirname(p1) == str(tmp_path) and "_b1_" in p1 and "_b4_" in p4
    fp1, fp4 = os.path.basename(p1).split("_")[1], os.path.basename(p4).split("_")[1]
    assert fp1 == fp4 and len(fp1) == 16
    assert SteerViTExtractor(prompts=["a"], backend="tensorrt").engine_path(1) != SteerViTExtractor(
        prompts=["a", "bb"], backend="tensorrt"
    ).engine_path(1)  # the prompts are part of the fingerprint
    other = _trt_node(engine_dir=str(tmp_path))
    with torch.no_grad():
        other._model.head.weight.add_(1.0)
    other._fingerprint = None
    assert other.engine_path(1) != p1


def test_tensorrt_import_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(ImportError, match="tensorrt-cu12"):
        trt_engine._tensorrt()
    monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="8.6.1"))
    with pytest.raises(ImportError, match=">= 10"):
        trt_engine._tensorrt()


def test_tensorrt_needs_cuda_names_the_build_command_and_checks_the_shape(
    fake_loader, fake_trt, tmp_path, monkeypatch
):
    node = _trt_node(engine_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="CUDA"):
        node._load_engine(1, torch.device("cpu"))
    with pytest.raises(FileNotFoundError, match="trt_engine build-pipeline"):
        node._load_engine(4, torch.device("cuda"))
    with pytest.raises(RuntimeError, match="CUDA"):
        node.build_engine(1)
    open(node.engine_path(4), "wb").close()
    monkeypatch.setattr(
        trt_engine,
        "TensorRTEngine",
        lambda p, d: types.SimpleNamespace(input_shape=(1, 3, RES, RES)),
    )
    with pytest.raises(RuntimeError, match="rebuild"):
        node._load_engine(4, torch.device("cuda"))


# ------------------------------------------------------------------ export and build
def test_export_leaves_the_node_model_as_it_was(fake_loader, tmp_path):
    pytest.importorskip("onnx")
    node = SteerViTExtractor(prompts=PROMPTS)
    before = {k: v.clone() for k, v in node._model.state_dict().items()}
    path = trt_engine.export_onnx(node._steered_pass(2), RES, str(tmp_path / "s.onnx"))
    assert os.path.getsize(path) > 0
    assert not node._model.training
    assert all(torch.equal(v, before[k]) for k, v in node._model.state_dict().items())


@pytest.mark.parametrize(
    "precision, flags", [("fp32", set()), ("tf32", {"TF32"}), ("fp16", {"TF32", "FP16"})]
)
def test_build_engine_writes_the_engine_and_its_record(
    fake_loader, fake_trt, monkeypatch, tmp_path, precision, flags
):
    monkeypatch.setattr(trt_engine, "export_onnx", lambda g, r, p: open(p, "wb").close() or p)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
    fake_trt.Builder.configs.clear()
    node = SteerViTExtractor(prompts=PROMPTS)
    path = str(tmp_path / "e" / "x.engine")
    record = trt_engine.build_engine(node._steered_pass(4), RES, precision, path, "fp")
    assert open(path, "rb").read() == b"engine"
    assert json.loads(open(path + ".json").read()) == record
    assert record["input"] == [4, 3, RES, RES] and record["prompts"] == len(PROMPTS)
    assert record["precision"] == precision and record["fingerprint"] == "fp"
    assert fake_trt.Builder.configs[-1].flags == flags


def test_build_engine_failures(fake_loader, fake_trt, monkeypatch, tmp_path):
    monkeypatch.setattr(trt_engine, "export_onnx", lambda g, r, p: open(p, "wb").close() or p)
    graph = SteerViTExtractor(prompts=PROMPTS)._steered_pass(1)
    with pytest.raises(ValueError, match="precision"):
        trt_engine.build_engine(graph, RES, "int8", str(tmp_path / "a"), "fp")
    fake_trt.OnnxParser.ok = False
    try:
        with pytest.raises(RuntimeError, match="parse"):
            trt_engine.build_engine(graph, RES, "fp32", str(tmp_path / "a"), "fp")
    finally:
        fake_trt.OnnxParser.ok = True
    fake_trt.Builder.blob = None
    try:
        with pytest.raises(RuntimeError, match="build failed"):
            trt_engine.build_engine(graph, RES, "fp32", str(tmp_path / "a"), "fp")
    finally:
        fake_trt.Builder.blob = b"engine"
    monkeypatch.delattr(fake_trt.BuilderFlag, "FP16")
    with pytest.raises(RuntimeError, match="FP16"):
        trt_engine.build_engine(graph, RES, "fp16", str(tmp_path / "a"), "fp")


def test_node_build_engine_skips_an_existing_engine(fake_loader, monkeypatch, tmp_path):
    node = _trt_node(engine_dir=str(tmp_path))
    engine = str(tmp_path / "fp32_0123456789abcdef_b4_r28_gpu_trt10.engine")
    monkeypatch.setattr(node, "engine_path", lambda batch, device=None: engine)
    monkeypatch.setattr(node, "_model_device", lambda: torch.device("cuda"))
    calls = []
    monkeypatch.setattr(
        trt_engine, "build_engine", lambda *a: calls.append(a) or open(a[3], "wb").close()
    )
    monkeypatch.setattr(node, "_steered_pass", lambda batch: f"pass-{batch}")
    node._engines = {4: object()}
    assert node.build_engine(4) == (engine, True) and 4 not in node._engines
    assert calls[0][0] == "pass-4" and calls[0][2] == "fp32" and calls[0][3] == engine
    assert node.build_engine(4) == (engine, False) and len(calls) == 1
    assert node.build_engine(4, force=True) == (engine, True) and len(calls) == 2


def test_node_batches_follow_the_tilers(tmp_path):
    doc = {
        "nodes": [
            {
                "name": "tiler",
                "class_name": "cuvis_ai_steervit.node.tiling.ImageTiler",
                "hparams": {"tiles": 2},
            },
            {
                "name": "t1",
                "class_name": "cuvis_ai_steervit.node.steervit.SteerViTExtractor",
                "hparams": {"backend": "tensorrt"},
            },
            {
                "name": "t2",
                "class_name": "cuvis_ai_steervit.node.steervit.SteerViTExtractor",
                "hparams": {"backend": "tensorrt"},
            },
            {
                "name": "t3",
                "class_name": "cuvis_ai_steervit.node.steervit.SteerViTExtractor",
                "hparams": {},
            },
        ],
        "connections": [
            {"source": "stretch.outputs.normalized", "target": "t1.inputs.rgb_image"},
            {"source": "tiler.outputs.tiles", "target": "t2.inputs.rgb_image"},
            {"source": "tiler.outputs.tiles", "target": "t3.inputs.rgb_image"},
        ],
    }
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert trt_engine.node_batches(str(path)) == {"t1": 1, "t2": 4}


def test_build_pipeline_cli_builds_each_node_for_its_batch(
    fake_loader, monkeypatch, tmp_path, capsys
):
    import cuvis_ai_core.utils.restore as restore

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    built = []
    node = _trt_node()
    monkeypatch.setattr(
        node,
        "build_engine",
        lambda batch, force=False: built.append((batch, force)) or ("/e/x", True),
    )
    monkeypatch.setattr(trt_engine, "node_batches", lambda y: {"steer": 4} if y == "p.yaml" else {})
    seen = {}
    monkeypatch.setattr(
        restore,
        "restore_pipeline",
        lambda y, **kw: seen.update(y=y, **kw) or types.SimpleNamespace(nodes=[node, "x"]),
    )
    assert trt_engine.main(["build-pipeline", "p.yaml"]) == 0
    assert built == [(4, False)] and seen["weights_path"] == "p.pt" and seen["device"] == "cuda"
    assert "built: /e/x (steer, batch 4)" in capsys.readouterr().out
    trt_engine.main(["build-pipeline", "q.yaml"])
    assert "no backend='tensorrt'" in capsys.readouterr().out


# ------------------------------------------------------------------ real TensorRT
@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize(
    "kw, rel_tol", [({}, 1e-4), ({"tf32": True}, 5e-3), ({"autocast_dtype": "float16"}, 3e-2)]
)
def test_real_tensorrt_engine_matches_the_torch_backend(fake_loader, tmp_path, kw, rel_tol):
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")
    ref = SteerViTExtractor(prompts=PROMPTS, **kw).cuda()
    node = _trt_node(engine_dir=str(tmp_path), **kw).cuda()
    for b in (1, 4):
        path, built = node.build_engine(b)
        assert built and os.path.exists(path + ".json")
        rgb = _rgb(b, seed=b).cuda()
        got, want = node(rgb_image=rgb), ref(rgb_image=rgb)
        for port in ("features", "scores"):
            scale = float(want[port].abs().max())
            assert float((got[port] - want[port]).abs().max()) <= rel_tol * scale, (b, port)
    assert not node._model.training


def test_an_engine_built_for_another_device_is_reloaded(fake_loader, monkeypatch):
    node = _trt_node()
    stale = FakeEngine(node._steered_pass(1))
    stale.device = torch.device("cuda", 1)  # bound to a GPU the input does not live on
    node._engines[1] = stale
    loaded = []
    monkeypatch.setattr(
        node,
        "_load_engine",
        lambda batch, device: loaded.append(device) or FakeEngine(node._steered_pass(batch)),
    )
    node(rgb_image=_rgb(1))
    assert loaded == [torch.device("cpu")] and node._engines[1] is not stale
    assert not stale.inputs  # the stale engine never ran


def test_build_pipeline_cli_needs_a_cuda_gpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as info:
        trt_engine.main(["build-pipeline", "p.yaml"])
    assert info.value.code == 1 and "CUDA GPU" in capsys.readouterr().err
