"""cuvis-ai-steervit: SteerViT (prompt-steered DINOv2) features and zero-shot anomaly maps.

Importing the package registers the plugin's weight declarations (the SteerViT checkpoint, the
DINOv2 trunk and the RoBERTa-large text encoder, :mod:`cuvis_ai_steervit.weights`) with
cuvis-ai-core's model-weight registry, so the node and ``download-model`` in the same environment
share the mirror pins without a plugin manifest on disk.
"""

from cuvis_ai_core.data.model_weights import ModelWeights

from cuvis_ai_steervit.node.steervit import SteerViTExtractor
from cuvis_ai_steervit.node.stretch import JointPercentileStretch
from cuvis_ai_steervit.node.tiling import GridStitcher, ImageTiler
from cuvis_ai_steervit.weights import PLUGIN_NAME, WEIGHTS

ModelWeights.register(PLUGIN_NAME, WEIGHTS)

__all__ = [
    "PLUGIN_NAME",
    "WEIGHTS",
    "GridStitcher",
    "ImageTiler",
    "JointPercentileStretch",
    "SteerViTExtractor",
]
