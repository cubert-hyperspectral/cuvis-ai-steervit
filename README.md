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

Local development — bare manifest pointing at the checkout (path relative to the manifest):

```yaml
name: steervit
path: "../cuvis-ai-steervit"
package_name: cuvis-ai-steervit
capabilities:
  - class_name: cuvis_ai_steervit.node.steervit.SteerViTExtractor
  - class_name: cuvis_ai_steervit.node.stretch.JointPercentileStretch
```

Frozen install — replace `path` by `repo:` + `tag:` (see [`plugins.yaml`](plugins.yaml)). Pipelines
reference the plugin by its name in their `plugins:` list:

```yaml
plugins:
  - cuvis_ai_builtin
  - steervit
```

Dependencies are deliberately slim: `torch`, `timm<2` (DINOv2 trunk), `transformers>=4.57` (the
RoBERTa text tower; 5.x verified) and `huggingface_hub`. The first construction downloads the
checkpoint (~0.4 GB), the timm DINOv2 weights and `roberta-large` into the Hugging Face cache; an
offline deployment needs a warm cache or a local `checkpoint` path.

## Vendored upstream code

The five inference files of SteerViT live under `cuvis_ai_steervit/_vendor/steervit/` (MIT, see the
`NOTICE` there), pinned to the upstream commit recorded in [`.upstream-sync.yml`](.upstream-sync.yml);
the only local changes are a provenance header and package-relative imports. The upstream
distribution is not a dependency because it pins `transformers<5` and `pillow<12` and pulls the
training stack (jupyter, wandb, datasets, umap), none of which the inference path needs.
Re-sync with `python tools/sync_vendor.py <ref>`.

## Development

Tests run inside a cuvis-ai env (the plugin needs `cuvis-ai-core` + `cuvis-ai-schemas` + torch):

```bash
uv run --extra dev pytest tests -q                 # unit + integration (fake backbone, no network)
uv run --extra dev pytest tests -q -m slow         # real weights vs the golden reference (tests/data)
uv run --extra dev ruff check cuvis_ai_steervit tests
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.register_plugin('plugins.yaml'); print(r.list_plugins())"
```

`tests/data/steervit_golden.pt` was produced with the published checkpoint in the reference
environment (steervit 0.2.0, transformers 4.57, CPU); the slow test rebuilds the node in the
current environment and checks the prompted maps and features against it.

CI uses `uv run --no-sources --locked`; regenerate the lock with `uv lock --no-sources` after
dependency changes (the `[tool.uv.sources]` cu128 index is for local GPU syncs only).

## References

- Ruthardt, J., Gaur, M., Ramanan, D., Tapaswi, M., Asano, Y. M. *Steerable Visual Representations.*
  arXiv 2604.02327 (SteerViT). Code: github.com/manugaurdl/SteerViT (MIT). Weights:
  huggingface.co/JonaRuthardt/SteerViT (Apache-2.0).
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022 (PatchCore).

## License

Apache-2.0 — see [LICENSE](LICENSE). The vendored SteerViT files are MIT (see their `NOTICE`).
