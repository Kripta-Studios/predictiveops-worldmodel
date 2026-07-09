from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.data.datasets.legacy_mvtec import LegacyMvtecBottleAdapter  # noqa: E402


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
        "--output",
        type=Path,
        default=Path("outputs/data/legacy_mvtec_bottle_audit.json"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    adapter = LegacyMvtecBottleAdapter(
        manifest_csv=args.manifest_csv,
        data_root=args.data_root,
        validation_fraction=args.validation_fraction,
    )
    report = adapter.build_audit_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_id": report["manifest"]["dataset_id"],
                "records": report["manifest"]["records"],
                "leakage_passed": report["leakage_audit"]["passed"],
                "split_counts": report["manifest"]["split_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
