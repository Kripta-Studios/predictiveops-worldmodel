"""Strict claim gate for any state-of-the-art wording.

The project can be SOTA-target before this gate, but every required field below
must be explicitly true before stronger wording is allowed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REQUIRED_GATE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "protocol": ("official_split", "leakage_pass", "validation_only_selection"),
    "baselines": ("exact_strong_baselines", "simple_baselines"),
    "statistics": ("required_seeds_met", "confidence_intervals", "significance_test"),
    "generalization": ("held_out_asset_or_site", "cross_dataset_or_pilot"),
    "operations": (
        "lead_time_reported",
        "false_alarm_rate_reported",
        "latency_reported",
        "calibration_reported",
    ),
    "reproducibility": ("config_saved", "checksums_saved", "code_commit_saved"),
}


@dataclass(frozen=True)
class SotaGateFailure:
    field: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "reason": self.reason}


@dataclass(frozen=True)
class SotaGateResult:
    passed: bool
    failures: tuple[SotaGateFailure, ...]
    exemptions_used: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
            "exemptions_used": dict(self.exemptions_used),
        }


def _exemption_for(payload: Mapping[str, Any], full_field: str) -> str | None:
    exemptions = payload.get("exemptions", {})
    if not isinstance(exemptions, Mapping):
        return None
    value = exemptions.get(full_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def evaluate_sota_gate(payload: Mapping[str, Any]) -> SotaGateResult:
    """Evaluate a sota_gate.json-style mapping."""

    failures: list[SotaGateFailure] = []
    exemptions_used: dict[str, str] = {}
    for section, fields in REQUIRED_GATE_FIELDS.items():
        section_value = payload.get(section)
        if not isinstance(section_value, Mapping):
            for field in fields:
                full_field = f"{section}.{field}"
                exemption = _exemption_for(payload, full_field)
                if exemption is not None:
                    exemptions_used[full_field] = exemption
                    continue
                failures.append(
                    SotaGateFailure(field=full_field, reason="missing_section_or_field")
                )
            continue
        for field in fields:
            full_field = f"{section}.{field}"
            exemption = _exemption_for(payload, full_field)
            if exemption is not None:
                exemptions_used[full_field] = exemption
                continue
            value = section_value.get(field)
            if value is not True:
                reason = "missing" if field not in section_value else "not_true"
                failures.append(SotaGateFailure(field=full_field, reason=reason))
    return SotaGateResult(
        passed=not failures,
        failures=tuple(failures),
        exemptions_used=exemptions_used,
    )


def load_gate_payload(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("sota_gate.json must contain a JSON object.")
    return data


def evaluate_sota_gate_file(path: Path) -> SotaGateResult:
    return evaluate_sota_gate(load_gate_payload(path))
