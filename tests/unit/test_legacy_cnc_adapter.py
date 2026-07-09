from __future__ import annotations

import csv
from pathlib import Path

from forgeworld.data.contracts import ColumnRole
from forgeworld.data.datasets.legacy_cnc import LegacyCncWindowsAdapter


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "FileName": "P001_C1",
            "NumberOfCycle": "1",
            "ToolIndex": "1",
            "CycleToFailure": "10",
            "CycleToFailureNormalized": "0.5",
            "wear_class_name": "Healthy",
            "MillingToolType": "1",
            "ADOC": "10",
            "RDOC": "8.0",
            "HardnessMean": "40.0",
            "ToolHolderLength": "80",
            "ToolRotation": "3200",
            "FeedRate": "640",
            "ToolDiameter": "10",
            "split": "train",
        },
        {
            "FileName": "P002_C1",
            "NumberOfCycle": "1",
            "ToolIndex": "2",
            "CycleToFailure": "5",
            "CycleToFailureNormalized": "0.25",
            "wear_class_name": "Moderate",
            "MillingToolType": "1",
            "ADOC": "10",
            "RDOC": "8.0",
            "HardnessMean": "40.0",
            "ToolHolderLength": "80",
            "ToolRotation": "3200",
            "FeedRate": "640",
            "ToolDiameter": "10",
            "split": "val",
        },
        {
            "FileName": "P003_C1",
            "NumberOfCycle": "1",
            "ToolIndex": "3",
            "CycleToFailure": "2",
            "CycleToFailureNormalized": "0.1",
            "wear_class_name": "Worn",
            "MillingToolType": "1",
            "ADOC": "10",
            "RDOC": "8.0",
            "HardnessMean": "40.0",
            "ToolHolderLength": "80",
            "ToolRotation": "3200",
            "FeedRate": "640",
            "ToolDiameter": "10",
            "split": "test",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_legacy_cnc_adapter_classifies_static_settings_as_context(tmp_path: Path) -> None:
    manifest = tmp_path / "cnc_windows.csv"
    _write_manifest(manifest)
    adapter = LegacyCncWindowsAdapter(manifest)

    schema = adapter.classify_columns()

    assert schema.column("FeedRate").role is ColumnRole.CONTEXT
    assert schema.column("ToolRotation").role is ColumnRole.CONTEXT
    assert schema.column("CycleToFailure").role is ColumnRole.FORBIDDEN
    assert schema.column("wear_class_name").role is ColumnRole.OUTCOME
    assert schema.metadata["verified_action_columns"] == []


def test_legacy_cnc_adapter_audits_manifest_without_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "cnc_windows.csv"
    _write_manifest(manifest)
    adapter = LegacyCncWindowsAdapter(manifest)

    report = adapter.build_audit_report()

    assert report["timestamp_audit"]["passed"]
    assert report["split_manifest"]["groups_by_split"]["validation"] == ("2",)
    assert report["leakage_audit"]["passed"]
    assert report["manifest"]["records"] == 3
