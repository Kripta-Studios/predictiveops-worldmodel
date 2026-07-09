from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.baselines.legacy_mvtec import (  # noqa: E402
    LegacyMvtecBaselineConfig,
    run_legacy_mvtec_baselines,
)
from forgeworld.runtime.compute import detect_compute_profile, plan_workers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_root = ROOT.parent / "industrial_jepa_mvp" / "data"
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=default_data_root / "manifests" / "mvtec_bottle.csv",
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines/legacy_mvtec_bottle_simple"),
    )
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-percentile", type=float, default=95.0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-sets", default="color_stats,thumbnail_gray")
    parser.add_argument("--smoke", action="store_true", help="Use CPU smoke-mode worker settings.")
    parser.add_argument("--unsafe-debug", action="store_true")
    args = parser.parse_args()

    feature_sets = tuple(item.strip() for item in args.feature_sets.split(",") if item.strip())
    profile = detect_compute_profile()
    worker_plan = plan_workers(profile, smoke_mode=args.smoke)
    config = LegacyMvtecBaselineConfig(
        manifest_csv=args.manifest_csv,
        data_root=args.data_root,
        output_dir=args.output_dir,
        category=args.category,
        validation_fraction=args.validation_fraction,
        validation_percentile=args.validation_percentile,
        image_size=args.image_size,
        feature_sets=feature_sets,
        unsafe_debug=args.unsafe_debug,
    )
    report = run_legacy_mvtec_baselines(config, worker_plan)
    print(
        json.dumps(
            {
                "experiment_name": report["experiment_name"],
                "output_dir": str(args.output_dir),
                "split_counts": report["split_counts"],
                "runs": report["runs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
