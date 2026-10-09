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
are frozen (no Phase 1, no `TRAINABLE_BUFFERS`), read from cuvis-ai-core's model-weight registry at
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
| `hf_repo` | `cubert-gmbh/steervit` | Hugging Face repository the checkpoint is fetched from; the default (and `JonaRuthardt/SteerViT` at `4468b691…`, the upstream file it mirrors) resolves through the model-weight registry |
| `hf_revision` | `1a999b31…` (validated commit) | commit of `hf_repo` the checkpoint is fetched at; set it (or `null` for the default branch) together with another `hf_repo` |
| `prompts` | `["the anomaly in the object"]` | prompt ensemble of the zero-shot map (one backbone pass each); `"the anomaly in the <object>"` phrasings work best |
| `feature_prompt` | `null` | prompt steering `features`; `null` = `prompts[0]`; a prompt outside `prompts` costs one extra pass |
| `topk_frac` | 0.001 | pixel fraction averaged into `anomaly_score` |
| `score_activation` | `sigmoid` | `sigmoid` or `none` (raw logits) before averaging over prompts |
| `autocast_dtype` | `null` | `float16` / `bfloat16`: run the backbone and head under CUDA autocast |
| `tf32` | false | TF32 tensor-core matmuls in the float32 forward |
| `backend` | `torch` | `tensorrt`: run the backbone pass and head as a TensorRT engine (see below) |
| `engine_dir` | `null` | where the TensorRT engines are kept (default: the user cache) |

**TensorRT backend.** `backend: tensorrt` runs the text-conditioned backbone pass and the head as a
TensorRT engine with the node's cached prompt encodings baked in, one engine per batch size (a tiled
node sees its tiles as one batch); the preprocessing, prompt averaging, upsampling and score stay in
torch. The precision follows `autocast_dtype` / `tf32` (float16 -> fp16, `tf32` -> TF32, neither ->
IEEE float32). Build the engines once per machine:

```bash
pip install "cuvis-ai-steervit[tensorrt]"   # TensorRT 10 for torch's CUDA, onnx to build
python -m cuvis_ai_steervit.trt_engine build-pipeline pipeline.yaml
```

Engine file names carry a fingerprint of the weights and prompts, the precision, batch, resolution,
GPU and TensorRT version; they live in `$CUVIS_AI_TRT_ENGINE_DIR/steervit` (default
`~/.cache/cuvis-ai/tensorrt/steervit`) or `engine_dir`. Speed depends on the GPU. On Jetson Thor the
fp16 engine runs one 336 px frame in 3.1 ms (7.7 ms under autocast) and four tiles in 8.4 ms
(19.7 ms), while the TF32 engine is slower than PyTorch's TF32 path. TensorRT's fp16 moves the
features more than autocast does (median per-token error ~0.5 % vs ~0.1 %). In the walnut pipelines
the stand-rule decisions on 287 validation frames were unchanged; re-validate a pipeline before
switching it.

Under cuvis.next give deployed pipelines an explicit `engine_dir`: it runs each pipeline session with
its own empty home directory (`HOME` and `USERPROFILE`, on every platform), so the default folder has
no engines. Build the engines with the same yaml; `build-pipeline` writes them there.

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
RoBERTa text tower; 5.x verified) and `huggingface_hub`. The node is built from three files, declared
in `cuvis_ai_steervit/weights.py` and in the manifest's `weights:` block and served by cuvis-ai-core's
model-weight registry from byte-identical `cubert-gmbh` mirrors (commit-pinned, sha256-verified, no
token needed):

| registry name | mirror | size | licence |
|---|---|---|---|
| `steervit_dinov2_base` | `cubert-gmbh/steervit` (SteerViT checkpoint) | 93 MB | Apache-2.0 |
| `vit_base_patch14_dinov2_lvd142m` | `cubert-gmbh/vit_base_patch14_dinov2.lvd142m` (DINOv2 ViT-B/14 trunk) | 346 MB | Apache-2.0 |
| `roberta_large` | `cubert-gmbh/roberta-large` (text encoder and tokenizer files) | 1.42 GB | MIT |

Online, the first construction fetches them into the shared cache. An offline deployment
(`HF_HUB_OFFLINE=1`, e.g. a CuvisNEXT child environment) provisions them once:

```bash
uv run download-model download steervit_dinov2_base vit_base_patch14_dinov2_lvd142m roberta_large
```

A local `checkpoint` path skips the registry for the checkpoint; another `hf_repo` or revision is
downloaded from the hub.

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
  huggingface.co/JonaRuthardt/SteerViT (Apache-2.0), mirrored as huggingface.co/cubert-gmbh/steervit.
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022 (PatchCore).

## License

Apache-2.0 — see [LICENSE](LICENSE). The vendored SteerViT files are MIT (see their `NOTICE`).
