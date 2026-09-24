# Changelog

## [Unreleased]

### Added
- Added `SteerViTExtractor`: frozen SteerViT (DINOv2 ViT-B/14 with gated text cross-attention)
  emitting the prompt-steered patch tokens `features [B, G, G, D]`, the zero-shot prompted anomaly
  map `scores [B, H, W, 1]` (activation of the segmentation logits averaged over a prompt ensemble,
  one batched backbone pass per prompt) and the top-k `anomaly_score [B]`. Checkpoint fetched from
  the Hugging Face hub at construction; `feature_prompt` selects the prompt steering the features.
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
