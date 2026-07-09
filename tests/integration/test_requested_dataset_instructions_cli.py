from __future__ import annotations

import json
import subprocess
import sys


def test_requested_dataset_instructions_cli_outputs_json_for_swat() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/download/requested_dataset_instructions.py",
            "swat",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["dataset_id"] == "swat"
    assert payload["manual_request_required"] is True
    assert payload["official_source_url"].startswith("https://www.sutd.edu.sg/")
