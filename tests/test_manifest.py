"""Manifest loading: the checked-in plugins.yaml resolves through NodeRegistry (the skill's
required verification step) and every capability imports as a Node subclass."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry

from cuvis_ai_steervit.node.steervit import SteerViTExtractor
from cuvis_ai_steervit.node.stretch import JointPercentileStretch

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "plugins.yaml"


def test_manifest_registers_plugin_and_resolves_nodes():
    registry = NodeRegistry()
    registry.register_plugin(str(MANIFEST))
    assert registry.list_plugins() == ["steervit"]
    assert registry.get("SteerViTExtractor") is SteerViTExtractor
    assert registry.get("JointPercentileStretch") is JointPercentileStretch


def test_manifest_capabilities_are_importable_nodes():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["name"] == "steervit"
    assert manifest["package_name"] == "cuvis-ai-steervit"
    assert manifest["capabilities"], "capabilities must not be empty"
    for entry in manifest["capabilities"]:
        module_name, _, cls_name = entry["class_name"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), cls_name)
        assert issubclass(cls, Node), entry["class_name"]


def test_vendored_package_imports_without_the_upstream_distribution():
    """The five vendored files import through the package-relative path only."""
    mod = importlib.import_module("cuvis_ai_steervit._vendor.steervit")
    assert hasattr(mod, "SteerViT")
    text = (REPO / "cuvis_ai_steervit" / "_vendor" / "steervit" / "model.py").read_text("utf-8")
    assert "from steervit." not in text and "from .backbone import" in text
