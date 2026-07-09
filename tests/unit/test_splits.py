from __future__ import annotations

from forgeworld.data.splits import build_grouped_splits


def test_grouped_splits_keep_assets_isolated() -> None:
    records = [
        {"asset": f"a{asset}", "ts": f"2026-01-0{asset + 1}T00:00:00Z", "value": row}
        for asset in range(6)
        for row in range(2)
    ]

    manifest = build_grouped_splits(records, group_key="asset", time_key="ts")

    assert manifest.split_protocol == "grouped_chronological"
    assert manifest.validate_no_group_overlap()
    groups_by_split = manifest.groups_by_split
    assert set(groups_by_split) == {"train", "validation", "test"}
    assert set(groups_by_split["train"]).isdisjoint(groups_by_split["validation"])
    assert set(groups_by_split["train"]).isdisjoint(groups_by_split["test"])
    assert set(groups_by_split["validation"]).isdisjoint(groups_by_split["test"])


def test_grouped_splits_put_latest_groups_in_test() -> None:
    records = [
        {"asset": "early", "ts": "2026-01-01T00:00:00Z"},
        {"asset": "middle", "ts": "2026-02-01T00:00:00Z"},
        {"asset": "late", "ts": "2026-03-01T00:00:00Z"},
    ]

    manifest = build_grouped_splits(records, group_key="asset", time_key="ts")

    assert manifest.group_to_split["early"] == "train"
    assert manifest.group_to_split["middle"] == "validation"
    assert manifest.group_to_split["late"] == "test"
