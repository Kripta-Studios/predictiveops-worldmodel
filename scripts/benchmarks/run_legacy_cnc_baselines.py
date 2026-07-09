"""Run leakage-guarded simple baselines on the preserved legacy CNC dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.baselines.legacy_cnc import (  # noqa: E402
    LegacyCncBaselineConfig,
    run_legacy_cnc_baselines,
)
from forgeworld.runtime.compute import detect_compute_profile, plan_workers  # noqa: E402


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
        default=ROOT / "outputs" / "baselines" / "legacy_cnc_failure_soon",
    )
    parser.add_argument("--failure-horizon-cycles", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--smoke", action="store_true", help="Use CPU smoke-mode worker settings.")
    parser.add_argument("--unsafe-debug", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    profile = detect_compute_profile()
    worker_plan = plan_workers(profile, smoke_mode=args.smoke)
    config = LegacyCncBaselineConfig(
        raw_feature_csv=data_root / "raw" / "sensor" / "cnc_milling" / "FeatureAndMetadata_Milling.csv",
        manifest_csv=data_root / "manifests" / "cnc_windows.csv",
        output_dir=args.output_dir,
        failure_horizon_cycles=args.failure_horizon_cycles,
        seeds=seeds,
        unsafe_debug=args.unsafe_debug,
    )
    report = run_legacy_cnc_baselines(config, worker_plan)
    printable = {
        "experiment_name": report["experiment_name"],
        "split_counts": report["split_counts"],
        "seeds": report["seeds"],
        "summary": report["summary"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
