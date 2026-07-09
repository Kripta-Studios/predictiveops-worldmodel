from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from forgeworld.baselines.legacy_mvtec import (
    LegacyMvtecBaselineConfig,
    run_legacy_mvtec_baselines,
)
from forgeworld.runtime.compute import ComputeProfile, plan_workers


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(value, value, value)).save(path)


def _write_visual_fixture(root: Path) -> Path:
    rows: list[dict[str, str]] = []
    train_values = [10, 11, 12, 13, 14, 15]
    for index, value in enumerate(train_values):
        image = root / "train" / "good" / f"{index:03d}.png"
        _write_png(image, value)
        rows.append(
            {
                "split": "train",
                "category": "bottle",
                "label": "0",
                "defect_type": "good",
                "image": str(image),
                "mask": "",
            }
        )
    for index, value in enumerate([11, 13]):
        image = root / "test" / "good" / f"{index:03d}.png"
        _write_png(image, value)
        rows.append(
            {
                "split": "test",
                "category": "bottle",
                "label": "0",
                "defect_type": "good",
                "image": str(image),
                "mask": "",
            }
        )
    for index, value in enumerate([230, 240]):
        image = root / "test" / "broken" / f"{index:03d}.png"
        mask = root / "ground_truth" / "broken" / f"{index:03d}_mask.png"
        _write_png(image, value)
        _write_png(mask, 255)
        rows.append(
            {
                "split": "test",
                "category": "bottle",
                "label": "1",
                "defect_type": "broken",
                "image": str(image),
                "mask": str(mask),
            }
        )

    manifest = root / "mvtec_bottle.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_legacy_mvtec_baseline_smoke(tmp_path: Path) -> None:
    manifest = _write_visual_fixture(tmp_path / "data")
    config = LegacyMvtecBaselineConfig(
        manifest_csv=manifest,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        validation_fraction=0.34,
        validation_percentile=95.0,
        image_size=8,
        feature_sets=("color_stats",),
    )
    plan = plan_workers(ComputeProfile(cpu_threads=4, system_ram_gb=8), smoke_mode=True)

    report = run_legacy_mvtec_baselines(config, plan)

    assert report["leakage_audit"]["passed"]
    assert report["split_counts"] == {"train": 4, "validation": 2, "test": 4}
    assert report["threshold_selection"] == "validation_normal_scores_only_no_test_label_tuning"
    assert report["runs"][0]["auroc"] == 1.0
    assert report["runs"][0]["recall"] == 1.0
    assert report["what_this_result_does_not_prove"]
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "split_manifest.json").exists()
