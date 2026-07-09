"""Print ForgeWorld's detected compute profile and worker plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.runtime.compute import detect_compute_profile, plan_workers, write_compute_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect hardware and print a worker plan.")
    parser.add_argument("--smoke", action="store_true", help="Force CPU-only smoke-mode settings.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Preferred accelerator.",
    )
    parser.add_argument("--cpu-workers", type=int, help="Override planned CPU worker count.")
    parser.add_argument("--write-report", help="Optional JSON report path.")
    args = parser.parse_args()

    profile = detect_compute_profile()
    plan = plan_workers(
        profile,
        smoke_mode=args.smoke,
        device_preference=args.device,
        requested_cpu_workers=args.cpu_workers,
    )
    payload = {"profile": profile.to_dict(), "plan": plan.to_dict()}
    if args.write_report:
        write_compute_report(Path(args.write_report), profile, plan)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
