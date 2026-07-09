from __future__ import annotations

import csv
from pathlib import Path

from forgeworld.baselines.legacy_cnc import (
    LegacyCncBaselineConfig,
    select_feature_columns,
    run_legacy_cnc_baselines,
)
from forgeworld.runtime.compute import ComputeProfile, plan_workers


def _write_raw_feature_table(path: Path) -> None:
    columns = [
        "sensor_mean",
        "sensor_std",
        "FileName",
        "NumberOfCycle",
        "SampleIndex",
        "TollIndex",
        "MillingToolType",
        "ADOC",
        "RDOC",
        "HardnessMean",
        "ToolHolderLength",
        "CycleToFailure",
        "CycleToFailureNormalized",
    ]
    rows = [
        [0.1, 1.0, "a1", 1, 0, 1, 1, 5, 8, 36, 80, 20, 1.0],
        [0.2, 1.1, "a2", 2, 0, 1, 1, 5, 8, 36, 80, 5, 0.2],
        [0.3, 1.2, "b1", 1, 0, 2, 1, 5, 8, 38, 80, 20, 1.0],
        [0.4, 1.3, "b2", 2, 0, 2, 1, 5, 8, 38, 80, 5, 0.2],
        [0.5, 1.4, "c1", 1, 0, 3, 1, 5, 8, 40, 80, 20, 1.0],
        [0.6, 1.5, "c2", 2, 0, 3, 1, 5, 8, 40, 80, 5, 0.2],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow([f"Column{i}" for i in range(1, len(columns) + 1)])
        writer.writerow(columns)
        writer.writerows(rows)


def _write_manifest(path: Path) -> None:
    columns = [
        "FileName",
        "NumberOfCycle",
        "ToolIndex",
        "CycleToFailure",
        "CycleToFailureNormalized",
        "wear_class_name",
        "MillingToolType",
        "ADOC",
        "RDOC",
        "HardnessMean",
        "ToolHolderLength",
        "ToolRotation",
        "FeedRate",
        "ToolDiameter",
        "split",
    ]
    rows = [
        ["a1", 1, 1, 20, 1.0, "Healthy", 1, 5, 8, 36, 80, 3200, 640, 10, "train"],
        ["a2", 2, 1, 5, 0.2, "Worn", 1, 5, 8, 36, 80, 3200, 640, 10, "train"],
        ["b1", 1, 2, 20, 1.0, "Healthy", 1, 5, 8, 38, 80, 3200, 640, 10, "val"],
        ["b2", 2, 2, 5, 0.2, "Worn", 1, 5, 8, 38, 80, 3200, 640, 10, "val"],
        ["c1", 1, 3, 20, 1.0, "Healthy", 1, 5, 8, 40, 80, 3200, 640, 10, "test"],
        ["c2", 2, 3, 5, 0.2, "Worn", 1, 5, 8, 40, 80, 3200, 640, 10, "test"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_feature_selection_excludes_forbidden_columns() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "sensor_mean": [1.0],
            "CycleToFailure": [2],
            "CycleToFailureNormalized": [0.1],
            "MillingToolType": [1],
        }
    )

    selected = select_feature_columns(df, "sensor_plus_context")

    assert "sensor_mean" in selected
    assert "MillingToolType" in selected
    assert "CycleToFailure" not in selected
    assert "CycleToFailureNormalized" not in selected


def test_legacy_cnc_baseline_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "FeatureAndMetadata_Milling.csv"
    manifest = tmp_path / "cnc_windows.csv"
    _write_raw_feature_table(raw)
    _write_manifest(manifest)
    config = LegacyCncBaselineConfig(
        raw_feature_csv=raw,
        manifest_csv=manifest,
        output_dir=tmp_path / "out",
        seeds=(0,),
        model_names=("majority",),
        feature_sets=("sensor_plus_context",),
    )
    plan = plan_workers(ComputeProfile(cpu_threads=4, system_ram_gb=8), smoke_mode=True)

    report = run_legacy_cnc_baselines(config, plan)

    assert report["leakage_audit"]["passed"]
    assert report["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert report["summary"][0]["model"] == "majority"
    assert (tmp_path / "out" / "metrics.json").exists()
