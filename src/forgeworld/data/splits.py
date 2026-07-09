"""Grouped split construction that avoids random row splits across lifecycles/assets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from forgeworld.data.timestamps import parse_timestamp


@dataclass(frozen=True)
class SplitAssignment:
    record_index: int
    split: str
    group: str

    def to_dict(self) -> dict[str, int | str]:
        return {"record_index": self.record_index, "split": self.split, "group": self.group}


@dataclass(frozen=True)
class SplitManifest:
    split_protocol: str
    group_key: str
    assignments: tuple[SplitAssignment, ...]
    group_to_split: Mapping[str, str]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(assignment.split for assignment in self.assignments))

    @property
    def groups_by_split(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {}
        for group, split in self.group_to_split.items():
            result.setdefault(split, []).append(group)
        return {split: tuple(sorted(groups)) for split, groups in result.items()}

    def validate_no_group_overlap(self) -> bool:
        seen: dict[str, str] = {}
        for assignment in self.assignments:
            previous = seen.setdefault(assignment.group, assignment.split)
            if previous != assignment.split:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_protocol": self.split_protocol,
            "group_key": self.group_key,
            "counts": self.counts,
            "groups_by_split": self.groups_by_split,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }


def _group_sort_value(
    records: list[Mapping[str, Any]], indices: list[int], time_key: str | None
) -> tuple[Any, str]:
    if time_key is None:
        return (str(records[indices[0]].get("__group__", "")), "")
    timestamps = [parse_timestamp(records[index][time_key]) for index in indices]
    return (min(timestamps), "")


def _split_group_counts(
    group_count: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("Grouped splits require at least three groups.")
    total_fraction = train_fraction + validation_fraction + test_fraction
    if total_fraction <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    train = round(group_count * train_fraction / total_fraction)
    validation = round(group_count * validation_fraction / total_fraction)
    train = max(1, min(train, group_count - 2))
    validation = max(1, min(validation, group_count - train - 1))
    test = group_count - train - validation
    if test <= 0:
        validation -= 1
        test = 1
    return train, validation, test


def build_grouped_splits(
    records: Iterable[Mapping[str, Any]],
    group_key: str,
    time_key: str | None = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitManifest:
    """Assign whole groups to train/validation/test splits.

    When ``time_key`` is provided, groups are ordered by their first timestamp and
    the latest groups become validation/test. This is a conservative default for
    lifecycle/process data and avoids random row leakage.
    """

    record_list = [dict(record) for record in records]
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(record_list):
        if group_key not in record:
            raise KeyError(f"Record {index} is missing group key {group_key!r}.")
        group = str(record[group_key])
        record["__group__"] = group
        grouped.setdefault(group, []).append(index)
    train_count, validation_count, _test_count = _split_group_counts(
        len(grouped), train_fraction, validation_fraction, test_fraction
    )
    ordered_groups = sorted(
        grouped,
        key=lambda group: _group_sort_value(record_list, grouped[group], time_key),
    )
    group_to_split: dict[str, str] = {}
    for position, group in enumerate(ordered_groups):
        if position < train_count:
            split = "train"
        elif position < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        group_to_split[group] = split
    assignments = tuple(
        SplitAssignment(record_index=index, split=group_to_split[group], group=group)
        for group, indices in grouped.items()
        for index in indices
    )
    return SplitManifest(
        split_protocol="grouped_chronological" if time_key else "grouped",
        group_key=group_key,
        assignments=assignments,
        group_to_split=group_to_split,
    )
