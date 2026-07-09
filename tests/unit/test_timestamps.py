from __future__ import annotations

from forgeworld.data.timestamps import validate_timestamps


def test_timestamp_audit_rejects_non_monotonic_group() -> None:
    records = [
        {"asset": "a", "ts": "2026-01-02T00:00:00Z"},
        {"asset": "a", "ts": "2026-01-01T00:00:00Z"},
    ]

    audit = validate_timestamps(records, "ts", ("asset",))

    assert not audit.passed
    assert audit.issues[0].code == "non_monotonic_timestamp"


def test_timestamp_audit_passes_monotonic_groups() -> None:
    records = [
        {"asset": "a", "ts": "2026-01-01T00:00:00Z"},
        {"asset": "b", "ts": "2025-01-01T00:00:00Z"},
        {"asset": "a", "ts": "2026-01-02T00:00:00Z"},
    ]

    audit = validate_timestamps(records, "ts", ("asset",))

    assert audit.passed
