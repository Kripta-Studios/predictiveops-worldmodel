from __future__ import annotations

from forgeworld.evaluation.sota_gate import REQUIRED_GATE_FIELDS, evaluate_sota_gate


def _passing_payload() -> dict[str, dict[str, bool]]:
    return {section: {field: True for field in fields} for section, fields in REQUIRED_GATE_FIELDS.items()}


def test_sota_gate_passes_only_when_all_required_fields_true() -> None:
    result = evaluate_sota_gate(_passing_payload())

    assert result.passed
    assert result.failures == ()


def test_sota_gate_fails_missing_and_false_fields() -> None:
    payload = _passing_payload()
    payload["protocol"]["leakage_pass"] = False
    del payload["operations"]["calibration_reported"]

    result = evaluate_sota_gate(payload)

    assert not result.passed
    failures = {failure.field: failure.reason for failure in result.failures}
    assert failures["protocol.leakage_pass"] == "not_true"
    assert failures["operations.calibration_reported"] == "missing"


def test_sota_gate_accepts_written_metric_specific_exemption() -> None:
    payload = _passing_payload()
    payload["operations"]["latency_reported"] = False
    payload["exemptions"] = {
        "operations.latency_reported": "Offline-only historical audit; no latency claim made."
    }

    result = evaluate_sota_gate(payload)

    assert result.passed
    assert result.exemptions_used["operations.latency_reported"].startswith("Offline-only")
