from __future__ import annotations

from pathlib import Path

import pytest

from forgeworld.data.datasets.legacy_cnc import LegacyCncWindowsAdapter


def test_local_legacy_cnc_manifest_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root.parent / "industrial_jepa_mvp" / "data" / "manifests" / "cnc_windows.csv"
    if not manifest.exists():
        pytest.skip("legacy Industrial JEPA MVP CNC manifest is not present")

    adapter = LegacyCncWindowsAdapter(manifest)
    report = adapter.build_audit_report()

    assert report["manifest"]["records"] > 0
    assert report["manifest"]["tool_groups"] >= 3
    assert report["timestamp_audit"]["passed"]
    assert report["split_manifest"]["split_protocol"] == "legacy_held_out_tool_manifest"
    assert report["leakage_audit"]["passed"]
    assert report["schema"]["metadata"]["verified_action_columns"] == []
