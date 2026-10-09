"""The plugin's weight declarations: valid rows, registered at import, provisioned with the plugin,
in step with the node's defaults and with the manifest's ``weights:`` block."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_core.data.model_weights import ModelWeights
from cuvis_ai_schemas.plugin import PluginWeightEntry

import cuvis_ai_steervit.node.steervit as mod
from cuvis_ai_steervit.weights import (
    DINOV2_TRUNK,
    PLUGIN_NAME,
    STEERVIT_CHECKPOINT,
    TEXT_ENCODER,
    WEIGHTS,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def test_rows_are_registered_at_import():
    assert [e.name for e in WEIGHTS] == [STEERVIT_CHECKPOINT, DINOV2_TRUNK, TEXT_ENCODER]
    for entry in WEIGHTS:
        row = ModelWeights.get(entry.name)
        assert row.entry == entry and row.plugin == PLUGIN_NAME
    assert ModelWeights.get("steervit_dinov2_base.pth").name == STEERVIT_CHECKPOINT


def test_every_row_is_provisioned_with_the_plugin():
    """The checkpoint is the default of its selector; the trunk and the text encoder have none."""
    assert all(ModelWeights.get(e.name).plugin_default for e in WEIGHTS)


def test_the_checkpoint_row_is_what_the_node_asks_for_by_default():
    entry = ModelWeights.get(STEERVIT_CHECKPOINT).entry
    assert (entry.repo_id, entry.revision, entry.filename) == (
        mod.DEFAULT_HF_REPO,
        mod.DEFAULT_HF_REVISION,
        mod.DEFAULT_CHECKPOINT,
    )
    assert entry.selected_by == "checkpoint" and entry.default


def test_the_mirrored_model_ids_point_at_declared_rows():
    names = {e.name for e in WEIGHTS}
    assert set(mod._MIRRORED_TRUNKS.values()) <= names
    assert set(mod._MIRRORED_TEXT_ENCODERS.values()) <= names


def test_the_manifest_weights_block_matches_the_declarations():
    """``emit_metadata`` writes the block from ``WEIGHTS``; a hand edit or stale run fails here."""
    manifest = yaml.safe_load((REPO / "plugins.yaml").read_text(encoding="utf-8"))
    rows = [PluginWeightEntry.model_validate(row) for row in manifest["weights"]]
    assert rows == list(WEIGHTS)
