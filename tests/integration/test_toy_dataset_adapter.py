from __future__ import annotations

from typing import Any, Mapping

from forgeworld.data import (
    ColumnRole,
    ColumnSpec,
    DatasetSchema,
    LeakageAuditReport,
    SplitManifest,
    TimestampAudit,
    build_grouped_splits,
    run_leakage_audit,
    validate_timestamps,
)


class ToyAdapter:
    def __init__(self) -> None:
        self.records = [
            {
                "ts": f"2026-01-0{asset + 1}T00:00:00Z",
                "asset_id": f"asset_{asset}",
                "feed_override": 100.0,
                "spindle_current": 5.0 + asset,
                "failed_next_cycle": False,
            }
            for asset in range(3)
        ]

    def build_manifest(self) -> Mapping[str, Any]:
        return {
            "dataset_id": "toy",
            "version": "v1",
            "records": len(self.records),
            "raw_data_committed": False,
        }

    def validate_timestamps(self) -> TimestampAudit:
        return validate_timestamps(self.records, "ts", ("asset_id",))

    def classify_columns(self) -> DatasetSchema:
        return DatasetSchema(
            dataset_id="toy",
            version="v1",
            time_column="ts",
            columns=(
                ColumnSpec("ts", ColumnRole.TIMESTAMP, "datetime"),
                ColumnSpec("asset_id", ColumnRole.ID, "string"),
                ColumnSpec(
                    "feed_override",
                    ColumnRole.ACTION,
                    "float",
                    "%",
                    time_aligned=True,
                    mutable_by_controller=True,
                ),
                ColumnSpec("spindle_current", ColumnRole.OBSERVATION, "float", "A"),
                ColumnSpec("failed_next_cycle", ColumnRole.OUTCOME, "bool"),
            ),
            group_columns=("asset_id",),
            outcome_columns=("failed_next_cycle",),
        )

    def build_grouped_splits(self) -> SplitManifest:
        return build_grouped_splits(self.records, "asset_id", "ts")

    def run_leakage_audit(self) -> LeakageAuditReport:
        return run_leakage_audit(
            self.classify_columns(),
            model_input_columns=("feed_override", "spindle_current"),
            split_manifest=self.build_grouped_splits(),
        )


def test_toy_adapter_smoke_protocol() -> None:
    adapter = ToyAdapter()

    assert adapter.build_manifest()["records"] == 3
    assert adapter.validate_timestamps().passed
    adapter.classify_columns().validate_or_raise()
    assert adapter.build_grouped_splits().validate_no_group_overlap()
    assert adapter.run_leakage_audit().passed
