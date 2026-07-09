"""Timestamp validation utilities for industrial sequence records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TimestampIssue:
    code: str
    message: str
    record_index: int | None = None
    group: tuple[Any, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "record_index": self.record_index,
            "group": list(self.group) if self.group is not None else None,
        }


@dataclass(frozen=True)
class TimestampAudit:
    issues: tuple[TimestampIssue, ...]
    records_checked: int
    metadata: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "records_checked": self.records_checked,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata or {}),
        }


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        result = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        result = datetime.fromisoformat(normalized)
    else:
        raise TypeError(f"Unsupported timestamp value: {value!r}")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def validate_timestamps(
    records: Iterable[Mapping[str, Any]],
    time_key: str,
    group_keys: tuple[str, ...] = (),
) -> TimestampAudit:
    """Check parseability and monotonic ordering within each declared group."""

    issues: list[TimestampIssue] = []
    last_seen: dict[tuple[Any, ...], datetime] = {}
    count = 0
    for index, record in enumerate(records):
        count += 1
        group = tuple(record.get(key) for key in group_keys) if group_keys else ("__all__",)
        if time_key not in record:
            issues.append(
                TimestampIssue(
                    code="missing_timestamp",
                    message="Record is missing the declared timestamp key.",
                    record_index=index,
                    group=group,
                )
            )
            continue
        try:
            timestamp = parse_timestamp(record[time_key])
        except (TypeError, ValueError) as exc:
            issues.append(
                TimestampIssue(
                    code="invalid_timestamp",
                    message=str(exc),
                    record_index=index,
                    group=group,
                )
            )
            continue
        previous = last_seen.get(group)
        if previous is not None and timestamp < previous:
            issues.append(
                TimestampIssue(
                    code="non_monotonic_timestamp",
                    message="Timestamps must not move backward within a group.",
                    record_index=index,
                    group=group,
                )
            )
        last_seen[group] = timestamp
    return TimestampAudit(issues=tuple(issues), records_checked=count)


def validate_numeric_sequence(
    records: Iterable[Mapping[str, Any]],
    order_key: str,
    group_keys: tuple[str, ...] = (),
    *,
    require_strictly_increasing: bool = False,
) -> TimestampAudit:
    """Validate a non-wall-clock temporal order key such as cycle number.

    Some public industrial datasets expose only cycle or operation indices. This
    audit keeps that limitation explicit while still checking temporal order.
    """

    issues: list[TimestampIssue] = []
    last_seen: dict[tuple[Any, ...], float] = {}
    count = 0
    for index, record in enumerate(records):
        count += 1
        group = tuple(record.get(key) for key in group_keys) if group_keys else ("__all__",)
        if order_key not in record:
            issues.append(
                TimestampIssue(
                    code="missing_temporal_order",
                    message="Record is missing the declared temporal order key.",
                    record_index=index,
                    group=group,
                )
            )
            continue
        try:
            value = float(record[order_key])
        except (TypeError, ValueError) as exc:
            issues.append(
                TimestampIssue(
                    code="invalid_temporal_order",
                    message=str(exc),
                    record_index=index,
                    group=group,
                )
            )
            continue
        previous = last_seen.get(group)
        if previous is not None:
            moved_backward = value < previous
            repeated = require_strictly_increasing and value <= previous
            if moved_backward or repeated:
                issues.append(
                    TimestampIssue(
                        code="non_monotonic_temporal_order",
                        message="Temporal order key must be monotonic within a group.",
                        record_index=index,
                        group=group,
                    )
                )
        last_seen[group] = value
    return TimestampAudit(
        issues=tuple(issues),
        records_checked=count,
        metadata={"temporal_kind": "numeric_order_key", "order_key": order_key},
    )
