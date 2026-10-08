"""SteerViT — prompt-steered DINOv2 patch features and a zero-shot prompted anomaly map.

SteerViT (Ruthardt et al., arXiv 2604.02327) is a frozen DINOv2 ViT-B/14 whose blocks are steered
by a text prompt through gated cross-attention: the backbone is text-conditioned, so every prompt
costs one backbone pass, and a linear head on the patch tokens grounds the prompt as a per-patch
segmentation logit. One forward of this node yields both views the walnut work uses:

- ``features``: the steered patch tokens on their patch grid, ``[B, G, G, D]`` (``[B, 24, 24, 768]``
  at 336 px). Fed to a PatchCore memory bank (``PatchCoreDetector`` with ``standardize=False``)
  they give a semantic-textural anomaly detector that complements a spectral bank.
- ``scores``: the zero-shot prompted anomaly map, sigmoid of the segmentation logits averaged over
  the prompt ensemble and upsampled to the input size, plus its top-k ``anomaly_score``.

The prompts are fixed hyper-parameters, so the text tower (RoBERTa-large, 1.4 GB) runs once at
construction: its connector features and attention masks are cached as buffers and the tower is
dropped. Inference is the vision backbone alone, the pipeline ``.pt`` carries ~0.4 GB instead of
~1.8 GB, and the numerics are those of the original text-conditioned forward. The weights are
frozen (no ``TRAINABLE_BUFFERS``, no Phase 1); they are downloaded from the Hugging Face hub at
construction and then travel with the pipeline ``.pt`` like any pretrained node. Input is an RGB
frame in ``[0, 1]`` (e.g. a false-RGB projection after
:class:`~cuvis_ai_steervit.node.stretch.JointPercentileStretch`); the node resizes it to the
model resolution and applies the model's own normalisation.

``backend="tensorrt"`` runs the text-conditioned backbone pass and the head as a TensorRT engine
built on the machine (:mod:`cuvis_ai_steervit.trt_engine`); the preprocessing, the prompt
averaging, the upsampling and the score stay in torch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor, nn

DEFAULT_CHECKPOINT = "steervit_dinov2_base.pth"
DEFAULT_HF_REPO = "JonaRuthardt/SteerViT"
# The commit of DEFAULT_HF_REPO the plugin was validated with (the tests' golden reference).
DEFAULT_HF_REVISION = "4468b69138d397fd329df00e80093387c26c77b2"  # pragma: allowlist secret
DEFAULT_PROMPTS = ("the anomaly in the object",)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_ACTIVATIONS = ("sigmoid", "none")
_BACKENDS = ("torch", "tensorrt")
_AUTOCAST_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@contextmanager
def _tf32_matmul(enabled: bool) -> Iterator[None]:
    """Allow TF32 tensor-core matmuls inside the block and restore the process setting after it."""
    if not enabled:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _load_steervit(checkpoint: str, hf_repo: str, hf_revision: str | None) -> nn.Module:
    """Build the SteerViT model from a local checkpoint path or a Hugging Face filename.

    Kept at module level so tests can substitute a small stand-in without touching the network.
    """
    from cuvis_ai_steervit._vendor.steervit import SteerViT

    path = checkpoint
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=hf_repo, filename=checkpoint, revision=hf_revision)
    return SteerViT.from_pretrained(path)


def _normalization_constants(model: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean / std of the model's eval transform; ImageNet constants when it exposes none."""
    try:
        for t in model.get_transforms().transforms:
            if t.__class__.__name__ == "Normalize":
                return tuple(float(x) for x in t.mean), tuple(float(x) for x in t.std)
    except Exception:  # noqa: BLE001 - any failure means "no transform available"
        pass
    return _IMAGENET_MEAN, _IMAGENET_STD


def _drop_engines(module: nn.Module, _incompatible_keys: Any) -> None:
    """``load_state_dict`` post hook: new weights need their own TensorRT engines."""
    module._engines = {}
    module._fingerprint = None


@torch.no_grad()
def _encode_prompts(model: nn.Module, prompts: list[str]) -> tuple[Tensor, Tensor]:
    """The text side of ``SteerViT.forward`` for one prompt list, tokenised together.

    Returns the connector features ``[P, L, D]`` and the full attention mask ``[P, T + L]`` (image
    tokens always attended, padded text tokens masked), exactly as the original forward builds them
    per call — so caching them per prompt list reproduces its numerics.
    """
    tok = model.tokenizer(
        list(prompts), padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    tok = {k: v.to(model.text_model.device) for k, v in tok.items()}
    text = model.text_model(**tok).last_hidden_state
    text = model.connector(F.normalize(text, dim=-1))
    mask = tok["attention_mask"].bool()
    ones = torch.ones(
        mask.shape[0], int(model.num_img_tokens), dtype=torch.bool, device=mask.device
    )
    return text.float(), torch.cat((ones, mask), dim=-1)


class SteerViTExtractor(Node):
    """Prompt-steered SteerViT patch features and the zero-shot anomaly map of an RGB frame."""

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.TORCH})

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame [B, H, W, 3] in [0, 1], e.g. a stretched false-RGB projection; "
            "resized to the model resolution and normalised internally.",
        ),
    }
    OUTPUT_SPECS = {
        "features": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Prompt-steered patch tokens on the patch grid [B, G, G, D] "
            "([B, 24, 24, 768] for ViT-B/14 at 336 px), steered by `feature_prompt`; the input "
            "of a PatchCore memory bank.",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Zero-shot prompted anomaly map [B, H, W, 1]: activation of the "
            "segmentation logits averaged over `prompts`, bilinearly upsampled to the input size.",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level score [B]: mean of the top topk_frac pixels of `scores`.",
        ),
    }

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        hf_repo: str = DEFAULT_HF_REPO,
        hf_revision: str | None = DEFAULT_HF_REVISION,
        prompts: list[str] | tuple[str, ...] = DEFAULT_PROMPTS,
        feature_prompt: str | None = None,
        topk_frac: float = 0.001,
        score_activation: str = "sigmoid",
        autocast_dtype: str | None = None,
        tf32: bool = False,
        backend: str = "torch",
        engine_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the node, load the frozen weights and cache the prompt encodings.

        Parameters
        ----------
        checkpoint : local path of a SteerViT checkpoint, or its filename in ``hf_repo``.
        hf_repo : Hugging Face repository the checkpoint is downloaded from when not a local path.
        hf_revision : commit of ``hf_repo`` the checkpoint is downloaded at; the default is the
            validated commit of the default repository, so set it (or ``None`` for the repo's
            default branch) together with another ``hf_repo``. Unused for a local path.
        prompts : text prompts of the zero-shot map; the map is the average over them (each prompt
            is one text-conditioned backbone pass). ``"the anomaly in the <object>"`` phrasings
            work best.
        feature_prompt : prompt that steers the `features` output; ``None`` uses ``prompts[0]``.
            A prompt outside ``prompts`` costs one extra backbone pass.
        topk_frac : fraction of output pixels averaged into `anomaly_score`.
        score_activation : ``"sigmoid"`` (default) or ``"none"`` (raw logits) before averaging.
        autocast_dtype : ``None`` (float32, default), ``"float16"`` or ``"bfloat16"``: run the ViT
            backbone and the segmentation head under CUDA autocast (tensor cores). Applied on CUDA
            inputs only; the outputs stay float32. The numerics change slightly, so re-validate and
            re-calibrate a pipeline before switching it.
        tf32 : allow TF32 tensor-core matmuls in the float32 forward (inputs rounded to a 10-bit
            mantissa, float32 storage and accumulation). CUDA inputs only; set around this node's
            forward and restored afterwards; ignored when ``autocast_dtype`` is set. Numerics change
            slightly, so re-validate and re-calibrate a pipeline before switching it.
        backend : ``"torch"`` (default) runs the model in PyTorch. ``"tensorrt"`` runs a TensorRT
            engine of the same backbone pass and head (CUDA only), one engine per batch size (a
            tiled node sees its tiles as one batch). Its precision follows the options above:
            ``autocast_dtype="float16"`` -> fp16 engine, ``tf32`` -> TF32 engine, neither -> IEEE
            float32 engine. TensorRT's fp16 moves the features more than autocast does; re-validate
            and re-calibrate a pipeline, and re-time it on the target GPU. Engines are specific to
            one GPU, TensorRT version, weights and prompt set; build them once per machine:
            ``python -m cuvis_ai_steervit.trt_engine build-pipeline <pipeline.yaml>``. Needs the
            ``tensorrt`` package (the plugin's ``tensorrt`` extra). Not offered for ``bfloat16``,
            nor with a ``feature_prompt`` outside ``prompts`` (that costs a second pass).
        engine_dir : where the engines are kept (default:
            :func:`cuvis_ai_steervit.trt_engine.default_engine_dir`). Engine file names carry the
            precision, a fingerprint of the weights and prompts, the batch, the resolution, the GPU
            and the TensorRT version, so one directory can serve several machines and pipelines.
        """
        prompts = [str(p) for p in prompts]
        if not prompts or any(not p.strip() for p in prompts):
            raise ValueError(
                "SteerViTExtractor: prompts must be a non-empty list of non-empty strings"
            )
        if feature_prompt is not None and not str(feature_prompt).strip():
            raise ValueError("SteerViTExtractor: feature_prompt must be None or a non-empty string")
        if hf_revision is not None and not str(hf_revision).strip():
            raise ValueError("SteerViTExtractor: hf_revision must be None or a non-empty string")
        if not 0.0 < float(topk_frac) <= 1.0:
            raise ValueError(f"SteerViTExtractor: topk_frac must be in (0, 1], got {topk_frac}")
        if score_activation not in _ACTIVATIONS:
            raise ValueError(
                f"SteerViTExtractor: score_activation must be one of {_ACTIVATIONS}, "
                f"got {score_activation!r}"
            )
        if autocast_dtype is not None and autocast_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(
                f"SteerViTExtractor: autocast_dtype must be None or one of "
                f"{sorted(_AUTOCAST_DTYPES)}, got {autocast_dtype!r}"
            )
        if backend not in _BACKENDS:
            raise ValueError(
                f"SteerViTExtractor: backend must be one of {_BACKENDS}, got {backend!r}"
            )
        if backend == "tensorrt" and _AUTOCAST_DTYPES.get(autocast_dtype) is torch.bfloat16:
            raise ValueError(
                "SteerViTExtractor: backend='tensorrt' has no bfloat16 engine; use float16"
            )
        if (
            backend == "tensorrt"
            and feature_prompt is not None
            and str(feature_prompt) not in prompts
        ):
            raise ValueError(
                "SteerViTExtractor: backend='tensorrt' needs feature_prompt among prompts "
                "(one pass per call)"
            )
        if engine_dir is not None and not str(engine_dir).strip():
            raise ValueError("SteerViTExtractor: engine_dir must be None or a non-empty path")
        self.checkpoint = str(checkpoint)
        self.hf_repo = str(hf_repo)
        self.hf_revision = str(hf_revision) if hf_revision is not None else None
        self.prompts = prompts
        self.feature_prompt = str(feature_prompt) if feature_prompt is not None else None
        self.topk_frac = float(topk_frac)
        self.score_activation = str(score_activation)
        self.autocast_dtype = autocast_dtype
        self.tf32 = bool(tf32)
        self.backend = str(backend)
        self.engine_dir = str(engine_dir) if engine_dir is not None else None
        super().__init__(
            checkpoint=self.checkpoint,
            hf_repo=self.hf_repo,
            hf_revision=self.hf_revision,
            prompts=list(self.prompts),
            feature_prompt=self.feature_prompt,
            topk_frac=self.topk_frac,
            score_activation=self.score_activation,
            autocast_dtype=self.autocast_dtype,
            tf32=self.tf32,
            backend=self.backend,
            engine_dir=self.engine_dir,
            **kwargs,
        )
        # TensorRT engines (one per batch size) load at the first forward of that batch, once the
        # pipeline has loaded the weights, and are dropped whenever new weights are loaded.
        self._engines: dict[int, Any] = {}
        self._fingerprint: str | None = None
        self.register_load_state_dict_post_hook(_drop_engines)
        self._amp_dtype = _AUTOCAST_DTYPES.get(autocast_dtype) if autocast_dtype else None

        model = _load_steervit(self.checkpoint, self.hf_repo, self.hf_revision)
        model.eval()
        mean, std = _normalization_constants(model)
        # Preprocessing constants, not fitted state: kept out of the state_dict.
        self.register_buffer(
            "_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self._resolution = int(model.image_size[0])
        self._grid = self._resolution // int(model.patch_size)
        self._num_prefix = int(model.vision_model.trunk.num_prefix_tokens)

        # Prompt encodings are constants of this node: compute them once, keep them as buffers
        # (they travel in the .pt) and drop the text tower.
        feats, mask = _encode_prompts(model, self.prompts)
        self.register_buffer("_prompt_feats", feats)
        self.register_buffer("_prompt_mask", mask)
        fp = self.feature_prompt or self.prompts[0]
        self._feature_index = self.prompts.index(fp) if fp in self.prompts else -1
        if self._feature_index < 0:
            f2, m2 = _encode_prompts(model, [fp])
            self.register_buffer("_feature_feats", f2)
            self.register_buffer("_feature_mask", m2)
        del model.text_model
        model.tokenizer = None
        # Frozen by construction, whichever loader built the model: no gradients, and eval mode
        # is re-applied in `train()` so a training-stage pipeline cannot switch dropout on and
        # make the features non-deterministic.
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

    # ------------------------------------------------------------------ helpers
    def train(self, mode: bool = True) -> SteerViTExtractor:
        """Follow the pipeline's mode flag but keep the frozen SteerViT in eval mode."""
        super().train(mode)
        self._model.eval()
        return self

    @property
    def grid_size(self) -> int:
        """Patch-grid side ``G`` (24 for ViT-B/14 at 336 px)."""
        return self._grid

    # ------------------------------------------------------------------ TensorRT
    @property
    def engine_precision(self) -> str:
        """The TensorRT engine precision of this node's options: ``fp16``, ``tf32`` or ``fp32``."""
        if self._amp_dtype is torch.float16:
            return "fp16"
        return "tf32" if self.tf32 else "fp32"

    def _model_device(self) -> torch.device:
        return self._prompt_feats.device

    def _steered_pass(self, batch: int) -> nn.Module:
        from cuvis_ai_steervit.trt_engine import SteeredPass

        return SteeredPass(
            self._model, self._prompt_feats, self._prompt_mask, self._num_prefix, batch
        )

    def engine_path(self, batch: int, device: torch.device | str | int | None = None) -> str:
        """This machine's engine file for ``batch`` frames with the current weights and prompts."""
        from cuvis_ai_steervit import trt_engine

        if self._fingerprint is None:
            tensors = {f"model.{k}": v for k, v in self._model.state_dict().items()}
            tensors.update(prompt_feats=self._prompt_feats, prompt_mask=self._prompt_mask)
            self._fingerprint = trt_engine.fingerprint(
                tensors, f"SteerViT|{self._resolution}|{self._num_prefix}|{self._feature_index}"
            )
        return os.path.join(
            self.engine_dir or trt_engine.default_engine_dir(),
            trt_engine.engine_file_name(
                self.engine_precision, self._fingerprint, batch, self._resolution, device
            ),
        )

    def build_engine(self, batch: int, force: bool = False) -> tuple[str, bool]:
        """Build this node's engine for ``batch`` frames (CUDA); returns its path, built now."""
        from cuvis_ai_steervit import trt_engine

        device = self._model_device()
        if device.type != "cuda":
            raise RuntimeError(
                f"{self.name}: TensorRT engines are built on a CUDA device, the model is on "
                f"{device}"
            )
        path = self.engine_path(batch, device)
        if os.path.exists(path) and not force:
            return path, False
        trt_engine.build_engine(
            self._steered_pass(batch),
            self._resolution,
            self.engine_precision,
            path,
            self._fingerprint or "",
        )
        self._engines.pop(int(batch), None)
        return path, True

    def _load_engine(self, batch: int, device: torch.device) -> Any:
        from cuvis_ai_steervit import trt_engine

        if device.type != "cuda":
            raise RuntimeError(f"{self.name}: backend='tensorrt' needs a CUDA device, got {device}")
        path = self.engine_path(batch, device)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{self.name}: no TensorRT engine for a batch of {batch} with these weights and "
                f"prompts on this machine at {path}. Build it once with: "
                "python -m cuvis_ai_steervit.trt_engine build-pipeline <this pipeline's yaml>"
            )
        engine = trt_engine.TensorRTEngine(path, device)
        expected = (batch, 3, self._resolution, self._resolution)
        if tuple(engine.input_shape) != expected:
            raise RuntimeError(
                f"{self.name}: TensorRT engine {path} takes input {tuple(engine.input_shape)}, "
                f"the node needs {expected}; rebuild it."
            )
        return engine

    def _preprocess(self, rgb_image: Tensor) -> Tensor:
        """BHWC [0, 1] -> normalised BCHW at the model resolution (bicubic)."""
        x = rgb_image.permute(0, 3, 1, 2)
        x = F.interpolate(
            x, size=(self._resolution, self._resolution), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        return (x - self._mean) / self._std

    def _run(self, x: Tensor, feats: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """One text-conditioned pass per (image, prompt): patch tokens and segmentation logits.

        ``feats [P, L, D]`` / ``mask [P, T + L]`` are cached prompt encodings; images are repeated
        per prompt (image-major order) so a single batched call covers the whole ensemble.
        Returns ``patch [B, P, G*G, D]`` and ``logits [B, P, G*G]``.
        """
        b, p = x.shape[0], feats.shape[0]
        if self.backend == "tensorrt" and feats is self._prompt_feats:
            engine = self._engines.get(b)
            if engine is None or engine.device != x.device:  # built for another GPU: reload
                engine = self._engines[b] = self._load_engine(b, x.device)
            out = engine(x)  # reused buffers: copy what leaves the node
            return out["tokens"].float().clone(), out["logits"].float().clone()
        amp = self._amp_dtype is not None and x.is_cuda
        tf32 = self.tf32 and x.is_cuda and not amp
        with (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=amp),
            _tf32_matmul(tf32),
        ):
            tokens = self._model.vision_model(
                x.repeat_interleave(p, dim=0), feats.repeat(b, 1, 1), attn_mask=mask.repeat(b, 1)
            )
            logits = self._model.get_heatmap_logits(tokens)
        patch = tokens[:, self._num_prefix :, :].float()  # outputs stay float32 under autocast
        return patch.reshape(b, p, patch.shape[1], patch.shape[2]), logits.float().reshape(b, p, -1)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def forward(self, rgb_image: Tensor, **_: Any) -> dict[str, Tensor]:
        """Steered features, prompted anomaly map and image score for a batch of RGB frames."""
        b, h, w, _ = rgb_image.shape
        x = self._preprocess(rgb_image)
        patch, logits = self._run(x, self._prompt_feats, self._prompt_mask)
        act = torch.sigmoid(logits) if self.score_activation == "sigmoid" else logits
        grid = act.mean(dim=1).reshape(b, 1, self._grid, self._grid)
        up = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
        scores = up.permute(0, 2, 3, 1).contiguous()
        k = max(1, int(self.topk_frac * h * w))
        anomaly_score = torch.topk(up.reshape(b, -1), k, dim=1).values.mean(dim=1)

        if self._feature_index >= 0:
            feats = patch[:, self._feature_index]
        else:
            feats = self._run(x, self._feature_feats, self._feature_mask)[0][:, 0]
        features = feats.reshape(b, self._grid, self._grid, -1).contiguous()
        return {"features": features, "scores": scores, "anomaly_score": anomaly_score}
