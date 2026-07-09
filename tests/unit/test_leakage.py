from __future__ import annotations

import pytest

from forgeworld.data.contracts import ColumnRole, ColumnSpec, DatasetSchema
from forgeworld.data.leakage import run_leakage_audit


def _base_schema(extra_columns: tuple[ColumnSpec, ...] = ()) -> DatasetSchema:
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
            ColumnSpec("current", ColumnRole.OBSERVATION, "float", "A"),
            ColumnSpec("failure", ColumnRole.OUTCOME, "bool"),
            *extra_columns,
        ),
        group_columns=("asset_id",),
        outcome_columns=("failure",),
    )


def test_leakage_audit_passes_for_valid_features() -> None:
    report = run_leakage_audit(_base_schema(), ("feed_override", "current"))

    assert report.passed
    assert report.errors() == ()


def test_leakage_audit_rejects_future_like_input_name() -> None:
    schema = _base_schema((ColumnSpec("cycle_to_failure", ColumnRole.OBSERVATION, "int"),))

    report = run_leakage_audit(schema, ("current", "cycle_to_failure"))

    assert not report.passed
    assert "future_or_outcome_like_input_name" in {issue.code for issue in report.errors()}


def test_failed_leakage_audit_blocks_training_unless_unsafe_debug() -> None:
    report = run_leakage_audit(_base_schema(), ("failure",))

    with pytest.raises(RuntimeError):
        report.assert_training_allowed()

    unsafe = report.assert_training_allowed(unsafe_debug=True)
    assert unsafe.unsafe_debug_watermark is not None
    assert not unsafe.passed
