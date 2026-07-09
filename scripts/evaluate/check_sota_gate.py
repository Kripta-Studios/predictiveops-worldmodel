"""CLI wrapper for ForgeWorld claim-gate evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.evaluation.sota_gate import evaluate_sota_gate_file


def _resolve_gate_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_dir():
        path = path / "sota_gate.json"
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a ForgeWorld sota_gate.json file.")
    parser.add_argument("path", help="Path to sota_gate.json or an output directory containing it.")
    parser.add_argument("--write-report", help="Optional JSON path for the evaluated gate result.")
    args = parser.parse_args()

    result = evaluate_sota_gate_file(_resolve_gate_path(args.path))
    payload = result.to_dict()
    if args.write_report:
        report_path = Path(args.write_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
