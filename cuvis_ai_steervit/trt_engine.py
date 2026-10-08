"""TensorRT engines for SteerViTExtractor's text-conditioned ViT pass, built per machine.

What an engine covers is the node's model call for a fixed batch: the steered DINOv2 backbone with
the node's cached prompt encodings baked in, plus the linear segmentation head - static input
``input`` (``B x 3 x R x R``, already resized and normalised by the node) and outputs ``tokens``
(patch tokens, ``B x P x G*G x D``) and ``logits`` (``B x P x G*G``). The node keeps its own
preprocessing, the prompt averaging, the upsampling and the score in torch.

An engine is compiled by the installed TensorRT for THIS GPU and TensorRT version from the node's
weights and prompts, so its file name carries a fingerprint of them next to the precision, the
batch, the resolution, the GPU and the TensorRT version; a JSON file next to each engine records how
it was built. Several machines and pipelines can share one directory.

Precisions (the node derives its own from ``autocast_dtype`` / ``tf32``):

- ``fp32``: IEEE float32, TF32 disabled (within ~2e-4 of the float32 tokens). Slow where the GPU
  relies on TF32.
- ``tf32``: TF32 tensor cores allowed (TensorRT's default float build).
- ``fp16``: TensorRT's FP16 builder flag. The tokens move more than under PyTorch's autocast
  (walnut pipelines: ~0.5 % median per-token error vs ~0.1 %, and pinning LayerNorm / softmax /
  reductions to float32 does not close the gap), although the downstream frame scores moved only
  ~0.1 %; validate a pipeline's decisions before switching it. TensorRT 10 only.

Whether an engine is faster depends on the GPU: re-time a pipeline on its target device (on Jetson
Thor the TF32 engine is slower than PyTorch's TF32 path, the fp16 engine about 2.4x faster than
autocast).

Optional dependencies, not installed with the plugin: the TensorRT Python package matching torch's
CUDA (``tensorrt-cu12`` / ``tensorrt-cu13``, version 10) and, to build engines, ``onnx`` (the
``tensorrt`` extra). Build the engines of a pipeline once per machine, before its first run::

    python -m cuvis_ai_steervit.trt_engine build-pipeline pipeline.yaml [more.yaml ...]
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import time
from typing import Any

import torch
from torch import Tensor, nn

PRECISIONS = ("fp32", "tf32", "fp16")
ENGINE_DIR_ENV = "CUVIS_AI_TRT_ENGINE_DIR"
_LOGGER: Any = None


def _tensorrt() -> Any:
    try:
        import tensorrt
    except ImportError as exc:
        raise ImportError(
            "SteerViTExtractor backend='tensorrt' needs the TensorRT Python package "
            "(version 10) matching torch's CUDA: pip install tensorrt-cu12 (CUDA 12 torch) or "
            "tensorrt-cu13 (CUDA 13 torch), or the plugin's 'tensorrt' extra; building engines "
            "also needs onnx."
        ) from exc
    if int(str(tensorrt.__version__).split(".")[0]) < 10:
        raise ImportError(f"TensorRT >= 10 is required, found {tensorrt.__version__}.")
    return tensorrt


def _logger(trt: Any) -> Any:
    """One TensorRT logger per process (TensorRT keeps the first and warns about others)."""
    global _LOGGER
    if _LOGGER is None or _LOGGER[0] is not trt:
        _LOGGER = (trt, trt.Logger(trt.Logger.WARNING))
    return _LOGGER[1]


def gpu_tag(device: torch.device | str | int | None = None) -> str:
    """``<device name>-sm<capability>``, filesystem-safe (e.g. ``NVIDIA-Thor-sm110``)."""
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    return f"{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')}-sm{major}{minor}"


def fingerprint(tensors: dict[str, Tensor], extra: str = "") -> str:
    """First 16 hex digits of the MD5 of named tensors (names, dtypes, shapes, values).

    An identity check of the weights and prompt encodings an engine was built from, not a security
    measure.
    """
    digest = hashlib.md5(extra.encode("utf-8"))  # noqa: S324
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()[:16]


def engine_file_name(
    precision: str,
    fp: str,
    batch: int,
    resolution: int,
    device: torch.device | str | int | None = None,
) -> str:
    """``<precision>_<fingerprint>_b<batch>_r<resolution>_<gpu tag>_trt<version>.engine``."""
    version = _tensorrt().__version__
    return (
        f"{precision}_{fp}_b{int(batch)}_r{int(resolution)}_{gpu_tag(device)}_trt{version}.engine"
    )


def default_engine_dir() -> str:
    """``$CUVIS_AI_TRT_ENGINE_DIR/steervit`` or ``~/.cache/cuvis-ai/tensorrt/steervit``."""
    root = os.environ.get(ENGINE_DIR_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "cuvis-ai", "tensorrt"
    )
    return os.path.join(root, "steervit")


class SteeredPass(nn.Module):
    """The exported graph: the node's model call for a fixed batch, prompt encodings as constants.

    ``forward(x [B, 3, R, R]) -> (tokens [B, P, G*G, D], logits [B, P, G*G])``, exactly the node's
    ``_run``.
    """

    def __init__(
        self, model: nn.Module, feats: Tensor, mask: Tensor, num_prefix: int, batch: int
    ) -> None:
        super().__init__()
        self.model = model
        self.num_prefix = int(num_prefix)
        self.batch = int(batch)
        self.prompts = int(feats.shape[0])
        self.register_buffer("feats", feats.repeat(batch, 1, 1))
        self.register_buffer("mask", mask.repeat(batch, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.model.vision_model(
            x.repeat_interleave(self.prompts, dim=0), self.feats, attn_mask=self.mask
        )
        logits = self.model.get_heatmap_logits(tokens)
        patch = tokens[:, self.num_prefix :, :]
        return (
            patch.reshape(self.batch, self.prompts, patch.shape[1], patch.shape[2]),
            logits.reshape(self.batch, self.prompts, -1),
        )


def export_onnx(graph: SteeredPass, resolution: int, path: str) -> str:
    """Export a :class:`SteeredPass` to ONNX (opset 17; ``input`` -> ``tokens``, ``logits``).

    Traced from a CPU copy: the TorchScript exporter's constant folding mixes CPU constants with
    CUDA weights. The copy is traced in eval mode; the node's own model is not touched.
    """
    cpu = copy.deepcopy(graph).to("cpu").eval()
    example = torch.zeros(graph.batch, 3, int(resolution), int(resolution))
    with torch.no_grad():
        torch.onnx.export(
            cpu,
            (example,),
            path,
            input_names=["input"],
            output_names=["tokens", "logits"],
            opset_version=17,
            dynamo=False,
        )
    return path


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_engine(
    graph: SteeredPass, resolution: int, precision: str, engine_path: str, fp: str
) -> dict[str, Any]:
    """Export ``graph``, build and save a TensorRT engine for this GPU; returns its build record."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}.")
    trt = _tensorrt()
    if precision == "fp16" and not hasattr(trt.BuilderFlag, "FP16"):
        raise RuntimeError(
            f"TensorRT {trt.__version__} has no FP16 builder flag (dropped in TensorRT 11): "
            "build fp16 engines with TensorRT 10, or use precision 'fp32' / 'tf32'."
        )
    t0 = time.perf_counter()
    logger = _logger(trt)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(graph, resolution, os.path.join(tmp, "steervit.onnx"))
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the exported SteerViT pass: {errors}")
        config = builder.create_builder_config()
        if precision == "fp32":
            config.clear_flag(trt.BuilderFlag.TF32)
        elif precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT engine build failed ({precision}).")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(blob)
    record = {
        "engine": os.path.basename(engine_path),
        "precision": precision,
        "fingerprint": fp,
        "input": [graph.batch, 3, int(resolution), int(resolution)],
        "prompts": graph.prompts,
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "cuvis_ai_steervit": _version("cuvis-ai-steervit"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_tag": gpu_tag(),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    with open(f"{engine_path}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    return record


class TensorRTEngine:
    """Run a serialized TensorRT engine (static shapes, one input) on torch CUDA tensors.

    The engine runs on its own CUDA stream (TensorRT adds a host synchronisation to every call on
    the default stream), ordered after the work already queued on torch's current stream, and
    torch's current stream waits for it, so torch ops that follow see the results in order. The
    output tensors are reused buffers, overwritten by the next call - consume or copy them before
    calling again.
    """

    def __init__(self, path: str, device: torch.device | str = "cuda") -> None:
        trt = _tensorrt()
        self.path = path
        self.device = torch.device(device)
        dtypes = {
            getattr(trt, name): dtype
            for name, dtype in (
                ("float32", torch.float32),
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
            )
            if hasattr(trt, name)
        }
        self._runtime = trt.Runtime(_logger(trt))
        with open(path, "rb") as f, torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
            self._stream = torch.cuda.Stream(self.device)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT could not load {path}: an engine only runs on the GPU and TensorRT "
                "version it was built with - rebuild it on this machine "
                "(python -m cuvis_ai_steervit.trt_engine build-pipeline ...)."
            )
        self._context = self._engine.create_execution_context()
        inputs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        self.outputs: dict[str, Tensor] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = dtypes[self._engine.get_tensor_dtype(name)]
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = (dtype, shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self.outputs[name] = buf
                self._context.set_tensor_address(name, buf.data_ptr())
        if len(inputs) != 1:
            raise RuntimeError(f"{path}: expected one engine input, found {sorted(inputs)}.")
        self.input_name, (self.input_dtype, self.input_shape) = next(iter(inputs.items()))
        self._held: Tensor | None = None

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        x = x.to(device=self.device, dtype=self.input_dtype).contiguous()
        self._context.set_tensor_address(self.input_name, x.data_ptr())
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)  # x written, previous outputs consumed
        with torch.cuda.device(self.device):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for {self.path}.")
        current.wait_stream(self._stream)  # everything queued after this sees the outputs
        self._held = x  # the engine reads x asynchronously; keep it alive until the next call
        return self.outputs


def node_batches(pipeline_yaml: str) -> dict[str, int]:
    """Frames per call of each ``backend: tensorrt`` SteerViTExtractor of a pipeline yaml.

    A node fed by an ``ImageTiler`` with ``tiles=k`` sees its ``k * k`` tiles as one batch; any
    other node one frame. Only a direct connection from the tiler to ``rgb_image`` counts: with a
    node in between the engine is built for one frame, and the mismatch is reported when it loads.
    """
    import yaml

    with open(pipeline_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    nodes = {n["name"]: n for n in doc.get("nodes", [])}
    batches = {}
    for name, node in nodes.items():
        hp = node.get("hparams") or {}
        if (
            not str(node.get("class_name", "")).endswith("SteerViTExtractor")
            or hp.get("backend") != "tensorrt"
        ):
            continue
        batch = 1
        for c in doc.get("connections", []):
            if str(c.get("target")) == f"{name}.inputs.rgb_image":
                src = nodes.get(str(c.get("source")).split(".")[0], {})
                if str(src.get("class_name", "")).endswith("ImageTiler"):
                    batch = int((src.get("hparams") or {}).get("tiles", 1)) ** 2
        batches[name] = batch
    return batches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m cuvis_ai_steervit.trt_engine",
        description="Build the TensorRT engines of the backend='tensorrt' SteerViTExtractor "
        "nodes of pipelines.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    pipe = sub.add_parser(
        "build-pipeline", help="restore each pipeline with its weights and build its engines"
    )
    pipe.add_argument(
        "pipelines", nargs="+", help="pipeline yamls; the weights are the sibling .pt"
    )
    pipe.add_argument(
        "--plugins-dir", default=None, help="plugin catalog, only for yamls outside a cuvis-ai tree"
    )
    pipe.add_argument("--force", action="store_true", help="rebuild existing engines")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        ap.exit(
            1, "build-pipeline needs a CUDA GPU: engines are built on the device they run on.\n"
        )
    from cuvis_ai_core.utils.restore import restore_pipeline

    for yaml_path in args.pipelines:
        batches = node_batches(yaml_path)
        if not batches:
            print(f"{yaml_path}: no backend='tensorrt' SteerViTExtractor", flush=True)
            continue
        pipeline = restore_pipeline(
            yaml_path,
            weights_path=os.path.splitext(yaml_path)[0] + ".pt",
            device="cuda",
            plugins_dirs=[args.plugins_dir] if args.plugins_dir else None,
        )
        nodes = {n.name: n for n in pipeline.nodes if not isinstance(n, str)}
        for name, batch in batches.items():
            path, built = nodes[name].build_engine(batch, force=args.force)
            print(f"{'built' if built else 'exists'}: {path} ({name}, batch {batch})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
