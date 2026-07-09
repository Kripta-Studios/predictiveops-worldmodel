from __future__ import annotations

import csv
from pathlib import Path

from forgeworld.runtime.compute import ComputeProfile, plan_workers
from forgeworld.world_models.sensor_jepa import (
    LegacyCncWorldModelConfig,
    build_legacy_cnc_world_model_arrays,
    train_legacy_cnc_world_model,
)


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
    rows = []
    for tool in range(1, 4):
        split_offset = tool * 0.1
        for cycle in range(1, 13):
            ctf = 13 - cycle
            rows.append(
                [
                    cycle * 0.2 + split_offset,
                    cycle * 0.1 + split_offset,
                    f"t{tool}_c{cycle}",
                    cycle,
                    0,
                    tool,
                    1,
                    5,
                    8,
                    35 + tool,
                    80,
                    ctf,
                    ctf / 12,
                ]
            )
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
    rows = []
    splits = {1: "train", 2: "val", 3: "test"}
    for tool in range(1, 4):
        for cycle in range(1, 13):
            ctf = 13 - cycle
            rows.append(
                [
                    f"t{tool}_c{cycle}",
                    cycle,
                    tool,
                    ctf,
                    ctf / 12,
                    "Worn" if ctf <= 3 else "Healthy",
                    1,
                    5,
                    8,
                    35 + tool,
                    80,
                    3200,
                    640,
                    10,
                    splits[tool],
                ]
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_build_world_model_arrays_excludes_forbidden_inputs(tmp_path: Path) -> None:
    raw = tmp_path / "FeatureAndMetadata_Milling.csv"
    manifest = tmp_path / "cnc_windows.csv"
    _write_raw_feature_table(raw)
    _write_manifest(manifest)
    config = LegacyCncWorldModelConfig(
        raw_feature_csv=raw,
        manifest_csv=manifest,
        output_dir=tmp_path / "out",
        horizons=(1, 2),
        window_length=4,
        epochs=1,
    )

    arrays = build_legacy_cnc_world_model_arrays(config)

    assert "CycleToFailure" not in arrays.feature_names
    assert arrays.x.shape[1:] == (4, 2)
    assert set(arrays.horizons) == {1, 2}
    assert len(arrays.asset_id) == len(arrays.failure)
    assert len(arrays.cycle) == len(arrays.failure)


def test_train_world_model_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    raw = tmp_path / "FeatureAndMetadata_Milling.csv"
    manifest = tmp_path / "cnc_windows.csv"
    _write_raw_feature_table(raw)
    _write_manifest(manifest)
    config = LegacyCncWorldModelConfig(
        raw_feature_csv=raw,
        manifest_csv=manifest,
        output_dir=tmp_path / "out",
        horizons=(1,),
        window_length=4,
        epochs=1,
        batch_size=8,
        embedding_dim=8,
        hidden_dim=12,
        context_dim=6,
        horizon_dim=4,
    )
    plan = plan_workers(ComputeProfile(cpu_threads=4, system_ram_gb=8), smoke_mode=True)

    report = train_legacy_cnc_world_model(config, plan)

    assert report["status"] == "experimental"
    assert report["action_columns"] == []
    assert report["split_metrics"]["test"]["samples"] > 0
    assert "operational" in report["split_metrics"]["test"]
    assert (tmp_path / "out" / "checkpoint.pt").exists()
    assert (tmp_path / "out" / "metrics.json").exists()
