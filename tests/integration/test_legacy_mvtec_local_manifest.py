from __future__ import annotations

from pathlib import Path

import pytest

from forgeworld.data.datasets.legacy_mvtec import LegacyMvtecBottleAdapter


def test_local_legacy_mvtec_bottle_manifest_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root.parent / "industrial_jepa_mvp" / "data"
    manifest = data_root / "manifests" / "mvtec_bottle.csv"
    if not manifest.exists():
        pytest.skip("legacy Industrial JEPA MVP MVTec bottle manifest is not present")

    adapter = LegacyMvtecBottleAdapter(manifest, data_root)
    report = adapter.build_audit_report()

    assert report["manifest"]["records"] > 0
    assert report["manifest"]["split_counts"]["test"] > 0
    assert report["timestamp_audit"]["passed"]
    assert report["split_manifest"]["split_protocol"] == (
        "official_train_test_plus_train_normal_validation"
    )
    assert report["leakage_audit"]["passed"]
    assert report["schema"]["metadata"]["verified_action_columns"] == []
