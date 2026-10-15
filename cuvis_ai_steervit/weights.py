"""Weight declarations of the steervit plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_steervit/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so
CuvisNEXT and the installer know what to provision without importing the plugin.

``SteerViTExtractor`` builds its model from three files: a SteerViT checkpoint (the
prompt connector, the gated cross-attention layers and the segmentation head), the
DINOv2 trunk timm builds the backbone from, and the RoBERTa-large text encoder that
encodes the prompts.

Checkpoints, chosen by the node's ``checkpoint`` (and ``hf_repo``):

- ``steervit_dinov2_base``, the original base checkpoint (the default and the one the
  published pipelines were validated with), served from ``JonaRuthardt/SteerViT`` at its
  validated commit. Its upstream model card marks it for research use since 9 Oct 2026.
- ``steervit_dinov2_commercial_small`` and ``steervit_dinov2_commercial_base``, the
  checkpoints the authors publish for commercial use under Apache-2.0, from the unchanged
  mirror ``cubert-gmbh/steervit-commercial``. Their features differ from the original's,
  so a pipeline that switches refits its memory banks and re-validates.

The DINOv2 ViT-B/14 trunk and RoBERTa-large are byte-identical mirrors under
``cubert-gmbh`` of ``timm/vit_base_patch14_dinov2.lvd142m`` and ``FacebookAI/roberta-large``
(pins from cuvis-ai-core's ``tools/mirror_weights.py``). The small checkpoint's
ViT-S/14 trunk is not mirrored; timm downloads it from ``timm/vit_small_patch14_dinov2.lvd142m``.
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

PLUGIN_NAME = "steervit"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

STEERVIT_CHECKPOINT = "steervit_dinov2_base"
"""Registry name of the original SteerViT checkpoint (DINOv2 base variant)."""

STEERVIT_COMMERCIAL_SMALL = "steervit_dinov2_commercial_small"
"""Registry name of the commercial SteerViT checkpoint, DINOv2 small variant."""

STEERVIT_COMMERCIAL_BASE = "steervit_dinov2_commercial_base"
"""Registry name of the commercial SteerViT checkpoint, DINOv2 base variant."""

COMMERCIAL_HF_REPO = "cubert-gmbh/steervit-commercial"
"""Unchanged mirror of the commercial checkpoints of ``JonaRuthardt/SteerViT`` (at 18469724)."""

COMMERCIAL_HF_REVISION = "8c2a0918038b0019d3bbf07e0883b40d24e4b3f6"  # pragma: allowlist secret
"""Commit of :data:`COMMERCIAL_HF_REPO` the registry pins."""

DINOV2_TRUNK = "vit_base_patch14_dinov2_lvd142m"
"""Registry name of the DINOv2 ViT-B/14 trunk in timm's layout."""

TEXT_ENCODER = "roberta_large"
"""Registry name of the RoBERTa-large text encoder with its tokenizer files."""

WEIGHTS: tuple[PluginWeightEntry, ...] = (
    PluginWeightEntry(
        name=STEERVIT_CHECKPOINT,
        display_name="SteerViT (DINOv2 base, research)",
        summary="Prompt-steered features and zero-shot anomaly maps",
        used_for=["Anomaly detection", "Zero-shot", "Text prompts"],
        repo_id="JonaRuthardt/SteerViT",
        filename="steervit_dinov2_base.pth",
        revision="4468b69138d397fd329df00e80093387c26c77b2",  # pragma: allowlist secret
        sha256="3ee74e5fba5cdbcbbdc35042b59fc44d54a68864724186c0ba4f5e2174892e3a",
        size_bytes=92_849_183,
        license="other",
        license_file=None,
        aliases=["steervit_dinov2_base.pth"],
        selected_by="checkpoint",
        default=True,
        description=(
            "Original SteerViT checkpoint, DINOv2 base variant: the prompt connector, the "
            "gated cross-attention layers and the segmentation head (JonaRuthardt/SteerViT "
            "at 4468b691). The upstream model card marks it for research use since "
            "9 Oct 2026; for a product use a commercial checkpoint."
        ),
    ),
    PluginWeightEntry(
        name=STEERVIT_COMMERCIAL_SMALL,
        display_name="SteerViT commercial (DINOv2 small)",
        summary="Prompt-steered features, commercial use (Apache-2.0)",
        used_for=["Anomaly detection", "Zero-shot", "Text prompts"],
        repo_id=COMMERCIAL_HF_REPO,
        filename="steervit_dinov2_commercial_small.pth",
        revision=COMMERCIAL_HF_REVISION,
        sha256="6c23756039ebddb01c262710cb2f9ea774e905a6e134946b3176dbcc96c969ee",
        size_bytes=48_284_373,
        license="Apache-2.0",
        license_file="LICENSE",
        aliases=["steervit_dinov2_commercial_small.pth"],
        selected_by="checkpoint",
        description=(
            "Commercial SteerViT checkpoint, DINOv2 small variant (ViT-S/14 trunk, 384-d "
            "features), published by the authors under Apache-2.0 (mirror of "
            "JonaRuthardt/SteerViT at 18469724, unmodified)."
        ),
    ),
    PluginWeightEntry(
        name=STEERVIT_COMMERCIAL_BASE,
        display_name="SteerViT commercial (DINOv2 base)",
        summary="Prompt-steered features, commercial use (Apache-2.0)",
        used_for=["Anomaly detection", "Zero-shot", "Text prompts"],
        repo_id=COMMERCIAL_HF_REPO,
        filename="steervit_dinov2_commercial_base.pth",
        revision=COMMERCIAL_HF_REVISION,
        sha256="f863021d5aaedbfd4b11d5be639936f56cfe3111ab90db6b8f9c60e40ee64833",
        size_bytes=92_345_957,
        license="Apache-2.0",
        license_file="LICENSE",
        aliases=["steervit_dinov2_commercial_base.pth"],
        selected_by="checkpoint",
        description=(
            "Commercial SteerViT checkpoint, DINOv2 base variant (ViT-B/14 trunk, 768-d "
            "features), published by the authors under Apache-2.0 (mirror of "
            "JonaRuthardt/SteerViT at 18469724, unmodified)."
        ),
    ),
    PluginWeightEntry(
        name=DINOV2_TRUNK,
        display_name="DINOv2 ViT-B/14 (timm, LVD-142M)",
        summary="Backbone the SteerViT extractor steers",
        used_for=["Backbone", "Anomaly detection"],
        repo_id="cubert-gmbh/vit_base_patch14_dinov2.lvd142m",
        filename="model.safetensors",
        revision="a1b5c7e574a045b35b4f904f82a937f24dc3fd3e",
        sha256="55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b",
        size_bytes=346_334_872,
        aux_files=[
            AuxFile(
                path="config.json",
                size_bytes=614,
                sha256="01050c8ec1abcd2fb980c33cae05468bde6aa6768dc1f8c8ba38f2e5f828f1a7",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "DINOv2 ViT-B/14 trunk (LVD-142M) in timm's layout; SteerViTExtractor builds "
            "its backbone from it (mirror of timm/vit_base_patch14_dinov2.lvd142m at "
            "4685c99d, unmodified)."
        ),
    ),
    PluginWeightEntry(
        name=TEXT_ENCODER,
        display_name="RoBERTa-large",
        summary="Text encoder for SteerViT prompts",
        used_for=["Text prompts"],
        repo_id="cubert-gmbh/roberta-large",
        filename="model.safetensors",
        revision="65303b7c13bf207ee6724df7dfdbb5ebda9d0144",
        sha256="047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38",
        size_bytes=1_421_700_479,
        aux_files=[
            AuxFile(
                path="config.json",
                size_bytes=482,
                sha256="82ba49810e6441735e696033ac2512ee09da555507b2917ac3865b202d592cc3",
            ),
            AuxFile(
                path="vocab.json",
                size_bytes=898_823,
                sha256="9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
            ),
            AuxFile(
                path="merges.txt",
                size_bytes=456_318,
                sha256="1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            ),
            AuxFile(
                path="tokenizer.json",
                size_bytes=1_355_863,
                sha256="847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
            ),
            AuxFile(
                path="tokenizer_config.json",
                size_bytes=25,
                sha256="994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
            ),
        ],
        license="MIT",
        license_file="LICENSE",
        description=(
            "RoBERTa-large text encoder with its config and tokenizer files; "
            "SteerViTExtractor encodes its prompts with it when the node is built "
            "(mirror of FacebookAI/roberta-large at 722cf37b, unmodified)."
        ),
    ),
)
"""Every weight the steervit node loads from the shared cache."""
