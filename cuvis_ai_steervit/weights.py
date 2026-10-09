"""Weight declarations of the steervit plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_steervit/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so
CuvisNEXT and the installer know what to provision without importing the plugin.

``SteerViTExtractor`` builds its model from three files: the SteerViT checkpoint (the
prompt connector, the gated cross-attention layers and the segmentation head), the
DINOv2 ViT-B/14 trunk timm builds the backbone from, and the RoBERTa-large text encoder
that encodes the prompts. The pins come from ``tools/mirror_weights.py`` in
cuvis-ai-core: the three repositories are byte-identical copies under ``cubert-gmbh``
of ``JonaRuthardt/SteerViT``, ``timm/vit_base_patch14_dinov2.lvd142m`` and
``FacebookAI/roberta-large``.
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

PLUGIN_NAME = "steervit"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

STEERVIT_CHECKPOINT = "steervit_dinov2_base"
"""Registry name of the SteerViT checkpoint (DINOv2 base variant)."""

DINOV2_TRUNK = "vit_base_patch14_dinov2_lvd142m"
"""Registry name of the DINOv2 ViT-B/14 trunk in timm's layout."""

TEXT_ENCODER = "roberta_large"
"""Registry name of the RoBERTa-large text encoder with its tokenizer files."""

WEIGHTS: tuple[PluginWeightEntry, ...] = (
    PluginWeightEntry(
        name=STEERVIT_CHECKPOINT,
        display_name="SteerViT (DINOv2 base)",
        summary="Prompt-steered features and zero-shot anomaly maps",
        used_for=["Anomaly detection", "Zero-shot", "Text prompts"],
        repo_id="cubert-gmbh/steervit",
        filename="steervit_dinov2_base.pth",
        revision="1a999b3147722af849e810c8c46f409b59ae1e7f",
        sha256="3ee74e5fba5cdbcbbdc35042b59fc44d54a68864724186c0ba4f5e2174892e3a",
        size_bytes=92_849_183,
        license="Apache-2.0",
        license_file="LICENSE",
        aliases=["steervit_dinov2_base.pth"],
        selected_by="checkpoint",
        default=True,
        description=(
            "SteerViT checkpoint, DINOv2 base variant: the prompt connector, the gated "
            "cross-attention layers and the segmentation head every SteerViTExtractor "
            "loads (mirror of JonaRuthardt/SteerViT at 4468b691, unmodified)."
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
