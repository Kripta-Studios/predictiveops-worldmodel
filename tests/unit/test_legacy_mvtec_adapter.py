from __future__ import annotations

import csv
from pathlib import Path

from forgeworld.data.contracts import ColumnRole
from forgeworld.data.datasets.legacy_mvtec import LegacyMvtecBottleAdapter


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "split": "train",
            "category": "bottle",
            "label": "0",
            "defect_type": "good",
            "image": "train/good/000.png",
            "mask": "",
        },
        {
            "split": "train",
            "category": "bottle",
            "label": "0",
            "defect_type": "good",
            "image": "train/good/001.png",
            "mask": "",
        },
        {
            "split": "train",
            "category": "bottle",
            "label": "0",
            "defect_type": "good",
            "image": "train/good/002.png",
            "mask": "",
        },
        {
            "split": "test",
            "category": "bottle",
            "label": "0",
            "defect_type": "good",
            "image": "test/good/000.png",
            "mask": "",
        },
        {
            "split": "test",
            "category": "bottle",
            "label": "1",
            "defect_type": "broken",
            "image": "test/broken/000.png",
            "mask": "ground_truth/broken/000_mask.png",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_legacy_mvtec_adapter_classifies_labels_and_masks_as_non_inputs(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "mvtec_bottle.csv"
    _write_manifest(manifest)
    adapter = LegacyMvtecBottleAdapter(manifest, tmp_path, validation_fraction=0.34)

    schema = adapter.classify_columns()

    assert schema.column("image_path").role is ColumnRole.OBSERVATION
    assert schema.column("category").role is ColumnRole.CONTEXT
    assert schema.column("label").role is ColumnRole.OUTCOME
    assert schema.column("defect_type").role is ColumnRole.OUTCOME
    assert schema.column("mask_path").role is ColumnRole.FORBIDDEN
    assert schema.metadata["verified_action_columns"] == []


def test_legacy_mvtec_adapter_builds_train_normal_validation_split(tmp_path: Path) -> None:
    manifest = tmp_path / "mvtec_bottle.csv"
    _write_manifest(manifest)
    adapter = LegacyMvtecBottleAdapter(manifest, tmp_path, validation_fraction=0.34)

    report = adapter.build_audit_report()

    assert report["timestamp_audit"]["passed"]
    assert report["split_manifest"]["split_protocol"] == (
        "official_train_test_plus_train_normal_validation"
    )
    assert report["split_manifest"]["counts"] == {
        "train": 2,
        "validation": 1,
        "test": 2,
    }
    assert report["leakage_audit"]["passed"]
