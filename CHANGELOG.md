# Changelog

## [Unreleased]

### Added
- Added `cuvis_ai_steervit.weights`: the three files `SteerViTExtractor` builds its model from
  (`steervit_dinov2_base`, `vit_base_patch14_dinov2_lvd142m`, `roberta_large`), pinned to the
  byte-identical `cubert-gmbh` mirrors by revision, sha256 and size. The package registers them with
  cuvis-ai-core's `ModelWeights` at import and the manifest lists them in its `weights:` block
  (`emit_metadata`), so `download-model` and CuvisNEXT provision them for an offline runtime.

### Changed
- `SteerViTExtractor` reads its checkpoint, the DINOv2 trunk and RoBERTa-large through the
  model-weight registry (the shared cache, sha256-verified, fetched anonymously when online; an
  offline runtime without them gets an error naming `download-model`) instead of downloading them
  from three upstream repositories at construction. `hf_repo` defaults to `cubert-gmbh/steervit` at
  `1a999b31`; `JonaRuthardt/SteerViT` at `4468b691`, which pipelines saved before record, resolves to
  the same registry entry, and any other repository or revision is still downloaded from the hub.
  The vendored files are unchanged: during construction the vendored backbone's `timm` is a stand-in
  that loads the trunk from the registry file (`pretrained_cfg_overlay`), and the text encoder loads
  from the mirror's snapshot folder. The bytes are the same, so the outputs are too.

## 0.3.0 - 2026-09-28

### Added
- Added a TensorRT backend to `SteerViTExtractor` (`backend="tensorrt"`, `engine_dir`): the
  text-conditioned backbone pass and the head run as a TensorRT engine with the node's cached
  prompt encodings baked in, one engine per batch size; the preprocessing, prompt averaging,
  upsampling and score stay in torch. The engine precision follows the node's options (float16
  autocast -> fp16, `tf32` -> TF32, neither -> IEEE float32). Engine file names carry a fingerprint
  of the weights and prompts, the precision, batch, resolution, GPU and TensorRT version; loading
  new weights drops the loaded engines. Not offered for bfloat16, nor with a `feature_prompt`
  outside `prompts`.
- Added `python -m cuvis_ai_steervit.trt_engine build-pipeline <yaml>`, which builds the engines of
  a pipeline's `backend: tensorrt` nodes, each for the batch it sees (an `ImageTiler` with `tiles=k`
  feeds `k*k` tiles).
- Added the `tensorrt` extra: TensorRT 10.15.1.29 for torch's CUDA (`tensorrt-cu12` / `-cu13`) and
  onnx.
- On Jetson Thor the fp16 engine runs one 336 px frame in 3.1 ms (7.7 ms under autocast) and four
  tiles in 8.4 ms (19.7 ms); the TF32 engine is slower than PyTorch's TF32 path there. TensorRT's
  fp16 moves the features more than autocast (median per-token error ~0.5 % vs ~0.1 %); in the
  walnut pipelines the stand-rule decisions on 287 validation frames were unchanged.
- Deployed pipelines should set `engine_dir`: cuvis.next runs each session with its own empty home
  directory, so the default engine folder (`~/.cache/cuvis-ai/tensorrt/<plugin>`) has no engines.
- A `backend: tensorrt` node reloads its engine when its input arrives on another device than
  the engine was built for; `build-pipeline` stops with a message when no CUDA GPU is available.

## 0.2.0 - 2026-09-28

### Added
- Added `autocast_dtype` (`float16` / `bfloat16`) to `SteerViTExtractor`: the ViT backbone and the
  segmentation head run under CUDA autocast on tensor cores; CUDA inputs only, outputs stay
  float32. On the walnut multi-scale gate (laptop RTX 4070) float16 halves the SteerViT time
  (72.8 -> 32.6 ms for t1 + t2, 84 -> 44 ms per frame) with the gate frame scores within 0.06 %
  and identical decisions on the probe frames; re-validate a pipeline before switching it.
- Added `tf32` to `SteerViTExtractor`: TF32 tensor-core matmuls in the float32 forward (float32
  storage and accumulation), set around the node's forward and restored afterwards; ignored under
  `autocast_dtype`. On the walnut multi-scale gate on Jetson Thor, TF32 for SteerViT and PatchCore
  cuts the frame from 190 to 59-60 ms with the gate frame scores within 0.02 %.

### Changed
- `JointPercentileStretch` sorts each frame once for both percentiles instead of once per
  percentile (6.8 -> 4 ms per frame on Jetson Thor); the outputs are bit-identical.

### Fixed
- Fixed the release workflow uploading uv's `dist/.gitignore` as a release asset
  (`default.gitignore` on v0.1.1): it now uploads the wheel and the sdist only.

## 0.1.1 - 2026-09-28

### Fixed
- Fixed every dependency resolution that reads the `cuda` group's index pins (`uv sync`,
  `uv run`, the release workflow), which failed with "conflicting indexes for package torch": the
  base torch requirement is declared once per index fork, as in cuvis-ai. The v0.1.0 release
  workflow failed on this; installing the v0.1.0 tag as a git or path dependency is not affected.

### Added
- Added a CI step that resolves the project with its index sources (`uv lock --dry-run`) and a
  guard test that the base requirements follow the fork markers.

## 0.1.0 - 2026-09-28

### Added
- Added `ImageTiler` and `GridStitcher`: split every image of a batch into T x T equal tiles
  stacked along the batch, and reassemble per-tile grids into one grid per image (image-major,
  row-major order). Stateless, differentiable reshapes: `ImageTiler -> SteerViTExtractor ->
  GridStitcher` yields a T times finer feature grid (48 x 48 for T = 2) for a multi-scale PatchCore
  feature bank. Generic nodes, planned to move to cuvis-ai core.
- Added `SteerViTExtractor`: frozen SteerViT (DINOv2 ViT-B/14 with gated text cross-attention)
  emitting the prompt-steered patch tokens `features [B, G, G, D]`, the zero-shot prompted anomaly
  map `scores [B, H, W, 1]` (activation of the segmentation logits averaged over a prompt ensemble,
  one batched backbone pass per prompt) and the top-k `anomaly_score [B]`. The checkpoint is fetched
  from the Hugging Face hub at construction, pinned to the validated commit (`hf_revision`), unless
  `checkpoint` is a local path; `feature_prompt` selects the prompt steering the features.
  The prompt encodings (text tower + connector) are computed once at construction and cached as
  buffers, and the text tower is dropped: inference runs the vision backbone only and the pipeline
  weights carry ~0.4 GB instead of ~1.8 GB, with the numerics of the original forward.
- Added `JointPercentileStretch`: per-frame percentile stretch to `[0, 1]` with the bounds taken
  jointly over all channels (or per channel), optional 8-bit truncation.
- Vendored the five SteerViT inference files (MIT) under `cuvis_ai_steervit/_vendor/steervit/` with
  a provenance header, package-relative imports and `NOTICE`; pinned in `.upstream-sync.yml`,
  re-synced by `tools/sync_vendor.py`.
- Added the local-path plugin manifest (`plugins.yaml`), tests with a fake backbone (golden outputs,
  prompt batching, port contract, frozen state-dict round-trip, manifest loading, pipeline reload
  smoke) and a slow real-weights parity test against a golden reference produced in the reference
  environment (`tests/data/steervit_golden.pt`).
- Added example pipelines under `examples/`: the zero-shot prompted map on a cu3s stream, and the
  two-bank fusion (a raw 61-band bank and a SteerViT feature bank, both cuvis-ai-patchcore
  `PatchCoreDetector`s, calibrated by `ScoreRangeNormalizer` on their `scores` port and averaged by
  `ScoreMapFusion`) with its Phase-1 trainrun.
- Added the `cuda` dependency group for local GPU development: torch and torchvision come from the
  cu128 index (cu130 on aarch64 Linux / Jetson). The pins are scoped to the group, so an
  environment that installs the plugin as a path or git dependency inherits none; a guard test
  checks this and that the committed lock (the CI lock) resolves torch from PyPI.
- Targets cuvis-ai-core >= 0.17.4 and cuvis-ai-schemas >= 0.12.0 on Python 3.11 – 3.13.
- Added CI on Python 3.11 and 3.13 for every pull request (stacked PRs included) and the weekly
  dependency compatibility audit against cuvis-ai-core v0.17.4.
