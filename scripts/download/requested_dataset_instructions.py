"""Print owner-respecting download/request instructions for known datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forgeworld.data.priority import PRIORITY_A_DATASETS, get_priority_dataset  # noqa: E402


def _instruction_payload(dataset_id: str) -> dict[str, object]:
    dataset = get_priority_dataset(dataset_id)
    payload = dataset.to_dict()
    payload["instructions"] = [
        "Use the official owner URL only; do not substitute an unverified mirror.",
        "Review and record license/terms before downloading or using commercially.",
        "Download or request access manually when the policy says manual_request_only.",
        "Record archive filename, access date, checksum, extraction command, and local path.",
        "Run the dataset adapter timestamp, split, and leakage audits before training.",
    ]
    if dataset.download_policy.startswith("manual_request_only"):
        payload["manual_request_required"] = True
        payload["instructions"].insert(1, "Complete the owner request form; do not automate it.")
    else:
        payload["manual_request_required"] = False
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        help="Dataset id such as swat, nasa_milling_wear, or priority_a for all entries.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    if args.dataset == "priority_a":
        payload: object = [_instruction_payload(dataset.dataset_id) for dataset in PRIORITY_A_DATASETS]
    else:
        payload = _instruction_payload(args.dataset)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif isinstance(payload, list):
        for item in payload:
            print(f"{item['dataset_id']}: {item['download_policy']} -> {item['official_source_url']}")
    else:
        print(f"Dataset: {payload['dataset_id']} ({payload['name']})")
        print(f"Status: {payload['status']}")
        print(f"Policy: {payload['download_policy']}")
        print(f"Official URL: {payload['official_source_url']}")
        print(f"Terms URL: {payload['terms_url']}")
        print("Instructions:")
        for instruction in payload["instructions"]:
            print(f"- {instruction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
