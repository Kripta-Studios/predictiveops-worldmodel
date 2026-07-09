"""Audit the preserved Industrial JEPA MVP CNC window manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.data.datasets.legacy_cnc import (  # noqa: E402
    LegacyCncWindowsAdapter,
    default_legacy_cnc_manifest_path,
)


def _default_manifest_path() -> Path:
    data_root = os.environ.get("FORGEWORLD_LEGACY_MVP_DATA_ROOT")
    if data_root:
        return Path(data_root) / "manifests" / "cnc_windows.csv"
    return default_legacy_cnc_manifest_path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=_default_manifest_path())
    parser.add_argument(
        "--write-report",
        type=Path,
        default=ROOT / "outputs" / "data" / "legacy_cnc_windows_audit.json",
    )
    parser.add_argument("--print-full", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()

    adapter = LegacyCncWindowsAdapter(args.manifest_csv)
    report_path = adapter.write_audit_report(args.write_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.print_full:
        printable = report
    else:
        printable = {
            "dataset_id": report["manifest"]["dataset_id"],
            "records": report["manifest"]["records"],
            "tool_groups": report["manifest"]["tool_groups"],
            "split_counts": report["manifest"]["split_counts"],
            "source_sha256": report["manifest"]["source_sha256"],
            "timestamp_audit_passed": report["timestamp_audit"]["passed"],
            "leakage_audit_passed": report["leakage_audit"]["passed"],
            "verified_action_columns": report["schema"]["metadata"]["verified_action_columns"],
            "report_path": str(report_path),
        }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if report["leakage_audit"]["passed"] and report["timestamp_audit"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
