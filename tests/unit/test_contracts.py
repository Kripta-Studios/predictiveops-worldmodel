from __future__ import annotations

import pytest

from forgeworld.data.contracts import ColumnRole, ColumnSpec, DatasetSchema, validate_model_inputs


def test_valid_action_requires_time_alignment_and_mutability() -> None:
    schema = DatasetSchema(
        dataset_id="toy",
        version="v1",
        time_column="ts",
        columns=(
            ColumnSpec("ts", ColumnRole.TIMESTAMP, "datetime"),
            ColumnSpec(
                "feed_setpoint",
                ColumnRole.ACTION,
                "float",
                "mm/min",
                time_aligned=True,
                mutable_by_controller=True,
            ),
            ColumnSpec("tool_id", ColumnRole.CONTEXT, "string"),
            ColumnSpec("spindle_current", ColumnRole.OBSERVATION, "float", "A"),
            ColumnSpec("failed_next_cycle", ColumnRole.OUTCOME, "bool"),
        ),
        group_columns=("tool_id",),
        outcome_columns=("failed_next_cycle",),
    )

    issues = schema.validate()

    assert not [issue for issue in issues if issue.severity == "error"]
    assert schema.features() == ("feed_setpoint", "tool_id", "spindle_current")


def test_metadata_misclassified_as_action_fails() -> None:
    schema = DatasetSchema(
        dataset_id="toy",
        version="v1",
        time_column="ts",
        columns=(
            ColumnSpec("ts", ColumnRole.TIMESTAMP, "datetime"),
            ColumnSpec("tool_id", ColumnRole.ACTION, "string", time_aligned=False),
        ),
    )

    issue_codes = {issue.code for issue in schema.validate()}

    assert "action_not_time_aligned" in issue_codes
    assert "action_not_mutable" in issue_codes


def test_outcome_cannot_be_model_input() -> None:
    schema = DatasetSchema(
        dataset_id="toy",
        version="v1",
        time_column="ts",
        columns=(
            ColumnSpec("ts", ColumnRole.TIMESTAMP, "datetime"),
            ColumnSpec("sensor", ColumnRole.OBSERVATION, "float", "A"),
            ColumnSpec("final_quality", ColumnRole.OUTCOME, "float"),
        ),
    )

    issues = validate_model_inputs(schema, ("sensor", "final_quality", "missing"))

    assert {issue.code for issue in issues} == {"disallowed_model_input", "unknown_model_input"}


def test_time_column_must_be_present_and_timestamp_role() -> None:
    schema = DatasetSchema(
        dataset_id="toy",
        version="v1",
        time_column="ts",
        columns=(ColumnSpec("ts", ColumnRole.ID, "datetime"),),
    )

    with pytest.raises(Exception):
        schema.validate_or_raise()
