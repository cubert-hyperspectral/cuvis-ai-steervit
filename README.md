# cuvis-ai-steervit

SteerViT for [cuvis-ai](https://docs.cuvis.ai/latest/): a frozen DINOv2 ViT-B/14 whose blocks are
steered by a text prompt through gated cross-attention (Ruthardt et al., *Steerable Visual
Representations*, arXiv 2604.02327). One forward pass of the node yields two views of an RGB frame:

- **prompt-steered patch features** `[B, 24, 24, 768]` — the input of a PatchCore memory bank
  (`PatchCoreDetector` from [cuvis-ai-patchcore](https://github.com/cubert-hyperspectral/cuvis-ai-patchcore)
  with `standardize: false`). Averaged with a bank on raw spectra this gives the walnut
  foreign-object detector that doubles the hard-object pixel AP of the spectral ensemble;
- **a zero-shot prompted anomaly map** `[B, H, W, 1]` — the paper's detector, no training at all:
  the sigmoid of the segmentation logits averaged over a prompt ensemble.

The backbone is text-conditioned, so every prompt is one backbone pass; the node batches the whole
ensemble into a single call. The prompts are fixed hyper-parameters, so the text tower
(RoBERTa-large) runs once at construction: its encodings are cached as buffers and the tower is
dropped, leaving the vision backbone (~0.4 GB) for inference and for the pipeline `.pt`. Weights
are frozen (no Phase 1, no `TRAINABLE_BUFFERS`), downloaded from the Hugging Face hub at
construction and stored in the pipeline `.pt` afterwards.

Requires `cuvis-ai-core >= 0.17.4` and `cuvis-ai-schemas >= 0.12.0` on Python 3.11 – 3.13.

## Nodes

### `cuvis_ai_steervit.node.steervit.SteerViTExtractor`

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `rgb_image` | in | `[B, H, W, 3]` float32 in `[0, 1]` | e.g. a false-RGB projection after `JointPercentileStretch`; resized (bicubic) to the model resolution and normalised internally |
| `features` | out | `[B, G, G, D]` float32 | steered patch tokens on the patch grid (`[B, 24, 24, 768]` at 336 px), steered by `feature_prompt` |
| `scores` | out | `[B, H, W, 1]` float32 | zero-shot prompted anomaly map, averaged over `prompts`, upsampled to the input size |
| `anomaly_score` | out | `[B]` float32 | mean of the top `topk_frac` pixels of `scores` |

| hparam | default | meaning |
|---|---|---|
| `checkpoint` | `steervit_dinov2_base.pth` | local checkpoint path, or its filename in `hf_repo` |
| `hf_repo` | `JonaRuthardt/SteerViT` | Hugging Face repository the checkpoint is fetched from |
| `hf_revision` | `4468b691…` (validated commit) | commit of `hf_repo` the checkpoint is fetched at; set it (or `null` for the default branch) together with another `hf_repo` |
| `prompts` | `["the anomaly in the object"]` | prompt ensemble of the zero-shot map (one backbone pass each); `"the anomaly in the <object>"` phrasings work best |
| `feature_prompt` | `null` | prompt steering `features`; `null` = `prompts[0]`; a prompt outside `prompts` costs one extra pass |
| `topk_frac` | 0.001 | pixel fraction averaged into `anomaly_score` |
| `score_activation` | `sigmoid` | `sigmoid` or `none` (raw logits) before averaging over prompts |

### `cuvis_ai_steervit.node.stretch.JointPercentileStretch`

Per-frame percentile stretch of a BHWC tensor to `[0, 1]`, with the percentiles taken jointly over
all channels (a per-channel stretch would re-balance the colour ratios). Stateless. `low` / `high`
(default 2 / 98) are the percentiles mapped to 0 / 1; `per_channel: true` switches to per-channel
bounds; `quantize_levels: 255` truncates to 8-bit levels exactly like a `uint8` image cast, which
is how the validated SteerViT feature bank was fed.

| Port | Direction | Shape / dtype |
|---|---|---|
| `data` | in | `[B, H, W, C]` float32 |
| `normalized` | out | `[B, H, W, C]` float32 in `[0, 1]` |

### `cuvis_ai_steervit.node.tiling.ImageTiler` / `cuvis_ai_steervit.node.tiling.GridStitcher`

Multi-scale helpers. `ImageTiler(tiles=T)` splits every image of a batch into a T x T grid of
equal, non-overlapping tiles and stacks them along the batch (tile `(i, j)` of image `b` at index
`b * T * T + i * T + j`), so a batched per-image node such as `SteerViTExtractor` runs on the tiles
unchanged and resizes each tile to its own resolution. `GridStitcher(tiles=T)` reassembles the
per-tile grids in the same order. With T = 2 the SteerViT patch grid becomes 48 x 48 instead of
24 x 24, so a small object no longer shares its patch with the surrounding context. Both nodes are
stateless reshapes (differentiable, device-agnostic); `ImageTiler` requires H and W divisible by T.
They are generic and planned to move to cuvis-ai core.

| Node | Port | Direction | Shape / dtype |
|---|---|---|---|
| `ImageTiler` | `image` | in | `[B, H, W, C]` float32 |
| `ImageTiler` | `tiles` | out | `[B * T * T, H / T, W / T, C]` float32 |
| `GridStitcher` | `tiles` | in | `[B * T * T, g_h, g_w, D]` float32 |
| `GridStitcher` | `grid` | out | `[B, T * g_h, T * g_w, D]` float32 |

A multi-scale feature bank runs the same prompt at two scales and averages the calibrated maps:

```
JointPercentileStretch ─┬─► SteerViTExtractor ─────────────────────────────► PatchCoreDetector ─► ScoreRangeNormalizer ─┐
                        └─► ImageTiler(2) ─► SteerViTExtractor ─► GridStitcher(2) ─► PatchCoreDetector ─► ScoreRangeNormalizer ─┴─► ScoreMapFusion (mean)
```

## Pipeline sketch: two memory banks on a cu3s stream

```
CU3SDataNode.cube ─┬─► PatchCoreDetector (61 bands, standardize) ──► ScoreRangeNormalizer ─┐
                   │                                                                        ├─► ScoreMapFusion (mean)
                   └─► FixedWavelengthSelector 640/550/470 ─► JointPercentileStretch          │
                          ─► SteerViTExtractor.features ─► PatchCoreDetector (768, standardize: false,
                                                             reference = cube) ─► ScoreRangeNormalizer ─┘
```

`PatchCoreDetector`, `ScoreRangeNormalizer` (each bank onto its normal range: fitted 1st / 99th
percentiles, floored, never clamped above) and `ScoreMapFusion` come from cuvis-ai-patchcore; the
selector is a cuvis-ai built-in. All fitted state comes from Phase 1 on normal frames. See `examples/`.

## Install

One manifest file is one plugin. For development, point it at a checkout (the path is relative to
the manifest file):

```yaml
name: steervit
path: "../cuvis-ai-steervit"
package_name: cuvis-ai-steervit
capabilities:
  - class_name: cuvis_ai_steervit.node.steervit.SteerViTExtractor
  - class_name: cuvis_ai_steervit.node.stretch.JointPercentileStretch
  - class_name: cuvis_ai_steervit.node.tiling.ImageTiler
  - class_name: cuvis_ai_steervit.node.tiling.GridStitcher
```

For a frozen, reproducible install, pin a release tag instead:

```yaml
name: steervit
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-steervit.git"
tag: "v0.1.0"
package_name: cuvis-ai-steervit
capabilities:
  - class_name: cuvis_ai_steervit.node.steervit.SteerViTExtractor
  - class_name: cuvis_ai_steervit.node.stretch.JointPercentileStretch
  - class_name: cuvis_ai_steervit.node.tiling.ImageTiler
  - class_name: cuvis_ai_steervit.node.tiling.GridStitcher
```

[`plugins.yaml`](plugins.yaml) is the local-path manifest of this repository, with the palette
metadata generated by cuvis-ai's `emit_metadata`. Pipelines reference the plugin by its name in
their `plugins:` list:

```yaml
plugins:
  - cuvis_ai_builtin
  - steervit
```

Dependencies are deliberately slim: `torch`, `timm<2` (DINOv2 trunk), `transformers>=4.57` (the
RoBERTa text tower; 5.x verified) and `huggingface_hub`. The first construction downloads three
files into the Hugging Face cache: the SteerViT checkpoint (93 MB, at `hf_revision`), the timm
DINOv2 ViT-B/14 weights (346 MB) and `roberta-large` (1.42 GB). An offline deployment
(`HF_HUB_OFFLINE=1`, e.g. a cuvis.next child environment) needs a warm cache holding all three, or
a local `checkpoint` path plus the two cached backbones.

## Vendored upstream code

The five inference files of SteerViT live under `cuvis_ai_steervit/_vendor/steervit/` (MIT, see the
`NOTICE` there), pinned to the upstream commit recorded in [`.upstream-sync.yml`](.upstream-sync.yml);
the only local changes are a provenance header and package-relative imports. The upstream
distribution is not a dependency because it pins `transformers<5` and `pillow<12` and pulls the
training stack (jupyter, wandb, datasets, umap), none of which the inference path needs.
Re-sync with `python tools/sync_vendor.py <ref>`.

## Development

CI runs the suite on Python 3.11 and 3.13 with the committed lock (a fake backbone, no network):

```bash
uv run --no-sources --locked --extra dev pytest tests/ -m "not slow"
uv run --no-sources --locked --extra dev pytest tests/ -m slow   # real weights vs the golden reference
uv run --no-sources --locked --extra dev ruff format --check cuvis_ai_steervit tests
uv run --no-sources --locked --extra dev ruff check cuvis_ai_steervit tests
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.register_plugin('plugins.yaml'); print(r.list_plugins())"
```

`tests/data/steervit_golden.pt` was produced with the published checkpoint in the reference
environment (steervit 0.2.0, transformers 4.57, CPU); the slow test rebuilds the node in the
current environment and checks the prompted maps and features against it.

A plain `uv sync` installs the `cuda` dependency group: torch and torchvision from the PyTorch cu128
index (cu130 on aarch64 Linux, e.g. Jetson). The pins are scoped to that group, so an environment
that installs the plugin as a path or git dependency inherits none of them. After a dependency
change, regenerate the lock with `uv lock --no-sources` (CI resolves torch from PyPI).

## References

- Ruthardt, J., Gaur, M., Ramanan, D., Tapaswi, M., Asano, Y. M. *Steerable Visual Representations.*
  arXiv 2604.02327 (SteerViT). Code: github.com/manugaurdl/SteerViT (MIT). Weights:
  huggingface.co/JonaRuthardt/SteerViT (Apache-2.0).
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022 (PatchCore).

## License

Apache-2.0 — see [LICENSE](LICENSE). The vendored SteerViT files are MIT (see their `NOTICE`).
