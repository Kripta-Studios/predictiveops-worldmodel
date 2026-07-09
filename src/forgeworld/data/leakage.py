"""Leakage audit utilities for prediction datasets and split manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from forgeworld.data.contracts import (
    FAIL_SEVERITIES,
    ColumnRole,
    DatasetSchema,
    SchemaIssue,
    validate_model_inputs,
)
from forgeworld.data.splits import SplitManifest


FORBIDDEN_INPUT_NAME_RE = re.compile(
    r"(^label$|^target$|future|lookahead|cycle[_-]?to[_-]?failure|"
    r"time[_-]?to[_-]?failure|remaining[_-]?useful[_-]?life|\brul\b|"
    r"final[_-]?quality|future[_-]?alarm|future[_-]?maintenance|post[_-]?failure)",
    flags=re.IGNORECASE,
)
FUTURE_AGGREGATE_RE = re.compile(
    r"(full[_-]?life|whole[_-]?lifecycle|global[_-]?(mean|max|min)|"
    r"all[_-]?cycles|expanding[_-]?)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LeakageIssue:
    code: str
    message: str
    severity: str = "error"
    column: str | None = None

    @classmethod
    def from_schema_issue(cls, issue: SchemaIssue) -> "LeakageIssue":
        return cls(
            code=issue.code,
            message=issue.message,
            severity=issue.severity,
            column=issue.column,
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "column": self.column,
        }


@dataclass(frozen=True)
class LeakageAuditReport:
    dataset_id: str
    passed: bool
    issues: tuple[LeakageIssue, ...]
    unsafe_debug_watermark: str | None = None

    def errors(self) -> tuple[LeakageIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity in FAIL_SEVERITIES)

    def assert_training_allowed(self, unsafe_debug: bool = False) -> "LeakageAuditReport":
        if self.passed:
            return self
        if unsafe_debug:
            return LeakageAuditReport(
                dataset_id=self.dataset_id,
                passed=False,
                issues=self.issues,
                unsafe_debug_watermark=(
                    "UNSAFE_DEBUG_LEAKAGE_AUDIT_FAILED_NOT_FOR_BENCHMARK_REPORTS"
                ),
            )
        failures = "; ".join(f"{issue.code}:{issue.column}" for issue in self.errors())
        raise RuntimeError(f"Leakage audit failed: {failures}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "passed": self.passed,
            "unsafe_debug_watermark": self.unsafe_debug_watermark,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _audit_suspicious_input_names(input_columns: Iterable[str]) -> tuple[LeakageIssue, ...]:
    issues: list[LeakageIssue] = []
    for name in input_columns:
        if FORBIDDEN_INPUT_NAME_RE.search(name):
            issues.append(
                LeakageIssue(
                    code="future_or_outcome_like_input_name",
                    message=(
                        "Model input name looks like a future, target, RUL, or lifecycle label."
                    ),
                    column=name,
                )
            )
        if FUTURE_AGGREGATE_RE.search(name):
            issues.append(
                LeakageIssue(
                    code="future_aggregate_like_input_name",
                    message="Model input name looks like an aggregate over future/lifecycle samples.",
                    column=name,
                )
            )
    return tuple(issues)


def run_leakage_audit(
    schema: DatasetSchema,
    model_input_columns: Iterable[str] | None = None,
    split_manifest: SplitManifest | None = None,
) -> LeakageAuditReport:
    """Audit schema, selected inputs, and split isolation before training."""

    inputs = tuple(model_input_columns) if model_input_columns is not None else schema.features()
    issues: list[LeakageIssue] = []
    issues.extend(LeakageIssue.from_schema_issue(issue) for issue in schema.validate())
    issues.extend(
        LeakageIssue.from_schema_issue(issue) for issue in validate_model_inputs(schema, inputs)
    )
    issues.extend(_audit_suspicious_input_names(inputs))
    for name in inputs:
        try:
            column = schema.column(name)
        except KeyError:
            continue
        if column.role in {ColumnRole.OUTCOME, ColumnRole.FORBIDDEN}:
            issues.append(
                LeakageIssue(
                    code="outcome_or_forbidden_input",
                    message="Outcome and forbidden columns cannot be model inputs.",
                    column=name,
                )
            )
    if split_manifest is not None and not split_manifest.validate_no_group_overlap():
        issues.append(
            LeakageIssue(
                code="group_overlap_between_splits",
                message="At least one group appears in multiple splits.",
            )
        )
    passed = not any(issue.severity in FAIL_SEVERITIES for issue in issues)
    return LeakageAuditReport(dataset_id=schema.dataset_id, passed=passed, issues=tuple(issues))
