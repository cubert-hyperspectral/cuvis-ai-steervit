# Changelog

## [Unreleased]

### Added
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
