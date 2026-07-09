"""Train the experimental deterministic CNC Sensor-JEPA world model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.runtime.compute import detect_compute_profile, plan_workers  # noqa: E402
from forgeworld.world_models.sensor_jepa import (  # noqa: E402
    LegacyCncWorldModelConfig,
    train_legacy_cnc_world_model,
)


def _default_data_root() -> Path:
    env_root = os.environ.get("FORGEWORLD_LEGACY_MVP_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return ROOT.parent / "industrial_jepa_mvp" / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "world_models" / "legacy_cnc_sensor_jepa_smoke",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, help="Run one seed only; overrides --seeds.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated development seeds.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizons", default="1,3")
    parser.add_argument("--smoke", action="store_true", help="Use CPU smoke-mode worker settings.")
    parser.add_argument("--unsafe-debug", action="store_true")
    args = parser.parse_args()

    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    seeds = (
        (args.seed,)
        if args.seed is not None
        else tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    )
    data_root = args.data_root
    profile = detect_compute_profile()
    worker_plan = plan_workers(profile, smoke_mode=args.smoke)
    smoke_scale = args.smoke
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        seed_output_dir = args.output_dir / f"seed_{seed}" if len(seeds) > 1 else args.output_dir
        config = LegacyCncWorldModelConfig(
            raw_feature_csv=(
                data_root / "raw" / "sensor" / "cnc_milling" / "FeatureAndMetadata_Milling.csv"
            ),
            manifest_csv=data_root / "manifests" / "cnc_windows.csv",
            output_dir=seed_output_dir,
            horizons=horizons,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embedding_dim=32 if smoke_scale else 64,
            hidden_dim=64 if smoke_scale else 128,
            context_dim=16 if smoke_scale else 32,
            horizon_dim=8 if smoke_scale else 16,
            unsafe_debug=args.unsafe_debug,
        )
        reports.append(train_legacy_cnc_world_model(config, worker_plan))
    aggregate = _aggregate_reports(reports, args.output_dir)
    printable = {
        "experiment_name": aggregate["experiment_name"],
        "status": aggregate["status"],
        "device": aggregate["device"],
        "seeds": aggregate["seeds"],
        "horizons": aggregate["horizons"],
        "test_summary": aggregate["test_summary"],
        "warnings": aggregate["warnings"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


def _summarize_metric(reports: list[dict[str, Any]], split: str, metric: str) -> dict[str, float | None]:
    values = [
        report["split_metrics"][split][metric]
        for report in reports
        if report["split_metrics"][split][metric] is not None
    ]
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
    }


def _aggregate_reports(reports: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "experiment_name": "legacy_cnc_deterministic_sensor_jepa_world_model_multiseed",
        "status": "experimental",
        "dataset": reports[0]["dataset"],
        "dataset_version": reports[0]["dataset_version"],
        "split_protocol": reports[0]["split_protocol"],
        "seeds": [report["seed"] for report in reports],
        "seed_count": len(reports),
        "device": reports[0]["device"],
        "horizons": reports[0]["horizons"],
        "claim_scope": "experimental_world_model_multiseed_not_claim_bearing",
        "test_set_influenced_model_selection": False,
        "threshold_selection": reports[0]["threshold_selection"],
        "test_summary": {
            metric: _summarize_metric(reports, "test", metric)
            for metric in ("forecast_mse", "failure_auroc", "failure_auprc", "failure_f1_at_0_5")
        },
        "validation_summary": {
            metric: _summarize_metric(reports, "validation", metric)
            for metric in ("forecast_mse", "failure_auroc", "failure_auprc", "failure_f1_at_0_5")
        },
        "warnings": sorted({warning for report in reports for warning in report["warnings"]}),
        "seed_reports": [
            {
                "seed": report["seed"],
                "checkpoint_path": report["checkpoint_path"],
                "test_metrics": report["split_metrics"]["test"],
                "final_train_loss": report["history"][-1]["loss"],
            }
            for report in reports
        ],
        "limitations": reports[0]["limitations"],
    }
    (output_dir / "multiseed_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return aggregate


if __name__ == "__main__":
    sys.exit(main())
