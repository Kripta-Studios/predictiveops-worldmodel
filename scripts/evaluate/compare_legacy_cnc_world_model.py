"""Compare the experimental CNC world model against recorded simple baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.evaluation.comparison import write_comparison_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-model-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "world_models"
        / "legacy_cnc_sensor_jepa_multiseed"
        / "multiseed_metrics.json",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "outputs" / "baselines" / "legacy_cnc_failure_soon_smoke" / "metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "comparisons" / "legacy_cnc_world_model_vs_baselines.json",
    )
    args = parser.parse_args()
    report = write_comparison_report(args.world_model_report, args.baseline_report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
