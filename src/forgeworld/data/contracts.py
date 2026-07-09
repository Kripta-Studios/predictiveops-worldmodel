"""Canonical data contracts and action/context/outcome validation.

The contract is deliberately explicit. It does not infer that a column is an
action from its name; dataset adapters must classify each column and this module
checks whether that classification is coherent with the AGENTS.md rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


class DatasetValidationError(ValueError):
    """Raised when a dataset contract violates a non-negotiable rule."""


class ColumnRole(StrEnum):
    """Allowed semantic roles for dataset columns."""

    ACTION = "action"
    CONTEXT = "context"
    OBSERVATION = "observation"
    OUTCOME = "outcome"
    ID = "id"
    TIMESTAMP = "timestamp"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


FAIL_SEVERITIES = frozenset({"error"})
FEATURE_INPUT_ROLES = frozenset(
    {ColumnRole.ACTION, ColumnRole.CONTEXT, ColumnRole.OBSERVATION}
)
NON_FEATURE_INPUT_ROLES = frozenset(
    {ColumnRole.OUTCOME, ColumnRole.FORBIDDEN, ColumnRole.ID, ColumnRole.TIMESTAMP}
)


@dataclass(frozen=True)
class SchemaIssue:
    """A structured schema validation issue."""

    code: str
    message: str
    severity: str = "error"
    column: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "column": self.column,
        }


@dataclass(frozen=True)
class ColumnSpec:
    """Machine-readable contract for one dataset column."""

    name: str
    role: ColumnRole | str
    dtype: str
    unit: str | None = None
    description: str = ""
    time_aligned: bool = False
    mutable_by_controller: bool = False
    timestamp_column: str | None = None
    allowed_as_model_input: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ColumnRole(self.role))
        if not self.name:
            raise DatasetValidationError("ColumnSpec.name cannot be empty.")
        if not self.dtype:
            raise DatasetValidationError(f"ColumnSpec.dtype cannot be empty for {self.name}.")

    @property
    def effective_allowed_as_model_input(self) -> bool:
        if self.allowed_as_model_input is not None:
            return self.allowed_as_model_input
        return self.role in FEATURE_INPUT_ROLES

    def validate(self) -> tuple[SchemaIssue, ...]:
        issues: list[SchemaIssue] = []
        if self.role is ColumnRole.ACTION:
            if not self.time_aligned:
                issues.append(
                    SchemaIssue(
                        code="action_not_time_aligned",
                        message="Action columns must be time-aligned commands or interventions.",
                        column=self.name,
                    )
                )
            if not self.mutable_by_controller:
                issues.append(
                    SchemaIssue(
                        code="action_not_mutable",
                        message=(
                            "Action columns must be controller/operator-changeable; "
                            "metadata belongs in context."
                        ),
                        column=self.name,
                    )
                )
        if self.role is ColumnRole.CONTEXT and self.mutable_by_controller:
            issues.append(
                SchemaIssue(
                    code="context_marked_mutable",
                    message=(
                        "Context columns should not be marked mutable by controller. "
                        "Use role=action only for time-aligned interventions."
                    ),
                    column=self.name,
                    severity="warning",
                )
            )
        if self.role in NON_FEATURE_INPUT_ROLES and self.allowed_as_model_input:
            issues.append(
                SchemaIssue(
                    code="non_feature_allowed_as_input",
                    message=f"{self.role.value} columns cannot be model features.",
                    column=self.name,
                )
            )
        if self.role is ColumnRole.OBSERVATION and self.unit is None:
            issues.append(
                SchemaIssue(
                    code="observation_missing_unit",
                    message="Observation columns should preserve physical units.",
                    column=self.name,
                    severity="warning",
                )
            )
        if self.role is ColumnRole.UNKNOWN and self.allowed_as_model_input:
            issues.append(
                SchemaIssue(
                    code="unknown_allowed_as_input",
                    message="Unknown semantic columns cannot be model features without mapping.",
                    column=self.name,
                )
            )
        return tuple(issues)


@dataclass(frozen=True)
class DatasetSchema:
    """Dataset-level schema with explicit column roles and grouping keys."""

    dataset_id: str
    version: str
    columns: tuple[ColumnSpec, ...]
    time_column: str
    id_columns: tuple[str, ...] = ()
    group_columns: tuple[str, ...] = ()
    outcome_columns: tuple[str, ...] = ()
    forbidden_columns: tuple[str, ...] = ()
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise DatasetValidationError("DatasetSchema.dataset_id cannot be empty.")
        if not self.version:
            raise DatasetValidationError("DatasetSchema.version cannot be empty.")
        if not self.time_column:
            raise DatasetValidationError("DatasetSchema.time_column cannot be empty.")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Unknown column: {name}")

    def features(self) -> tuple[str, ...]:
        return tuple(
            column.name for column in self.columns if column.effective_allowed_as_model_input
        )

    def validate(self) -> tuple[SchemaIssue, ...]:
        issues: list[SchemaIssue] = []
        names = self.column_names
        seen: set[str] = set()
        for name in names:
            if name in seen:
                issues.append(
                    SchemaIssue(
                        code="duplicate_column",
                        message="Column names must be unique.",
                        column=name,
                    )
                )
            seen.add(name)
        if self.time_column not in names:
            issues.append(
                SchemaIssue(
                    code="missing_time_column",
                    message="The declared time_column is not present in columns.",
                    column=self.time_column,
                )
            )
        else:
            time_role = self.column(self.time_column).role
            if time_role is not ColumnRole.TIMESTAMP:
                issues.append(
                    SchemaIssue(
                        code="time_column_role",
                        message="The declared time_column must have role=timestamp.",
                        column=self.time_column,
                    )
                )
        for column in self.columns:
            issues.extend(column.validate())
        for key in self.id_columns + self.group_columns + self.outcome_columns:
            if key not in names:
                issues.append(
                    SchemaIssue(
                        code="missing_declared_column",
                        message="A declared id/group/outcome column is not present in columns.",
                        column=key,
                    )
                )
        for name in self.outcome_columns:
            if name in names and self.column(name).role is not ColumnRole.OUTCOME:
                issues.append(
                    SchemaIssue(
                        code="outcome_role_mismatch",
                        message="Declared outcome columns must have role=outcome.",
                        column=name,
                    )
                )
        for name in self.forbidden_columns:
            if name in names and self.column(name).role is not ColumnRole.FORBIDDEN:
                issues.append(
                    SchemaIssue(
                        code="forbidden_role_mismatch",
                        message="Declared forbidden columns must have role=forbidden.",
                        column=name,
                    )
                )
        return tuple(issues)

    def validate_or_raise(self) -> None:
        issues = self.validate()
        errors = [issue for issue in issues if issue.severity in FAIL_SEVERITIES]
        if errors:
            formatted = "; ".join(f"{issue.code}:{issue.column}" for issue in errors)
            raise DatasetValidationError(formatted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "time_column": self.time_column,
            "id_columns": list(self.id_columns),
            "group_columns": list(self.group_columns),
            "outcome_columns": list(self.outcome_columns),
            "forbidden_columns": list(self.forbidden_columns),
            "source": self.source,
            "columns": [
                {
                    "name": column.name,
                    "role": column.role.value,
                    "dtype": column.dtype,
                    "unit": column.unit,
                    "description": column.description,
                    "time_aligned": column.time_aligned,
                    "mutable_by_controller": column.mutable_by_controller,
                    "timestamp_column": column.timestamp_column,
                    "allowed_as_model_input": column.effective_allowed_as_model_input,
                    "metadata": dict(column.metadata),
                }
                for column in self.columns
            ],
            "metadata": dict(self.metadata),
        }


def validate_model_inputs(schema: DatasetSchema, input_columns: Iterable[str]) -> tuple[SchemaIssue, ...]:
    """Validate that selected model inputs respect the schema roles."""

    issues: list[SchemaIssue] = []
    known = set(schema.column_names)
    for name in input_columns:
        if name not in known:
            issues.append(
                SchemaIssue(
                    code="unknown_model_input",
                    message="Model input column is not present in the dataset schema.",
                    column=name,
                )
            )
            continue
        column = schema.column(name)
        if not column.effective_allowed_as_model_input:
            issues.append(
                SchemaIssue(
                    code="disallowed_model_input",
                    message=f"{column.role.value} column cannot be used as a model input.",
                    column=name,
                )
            )
    return tuple(issues)
