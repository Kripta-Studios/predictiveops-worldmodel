"""Adapter for the preserved Industrial JEPA MVP CNC window manifest.

This adapter audits the legacy manifest as evidence, not as a claim-bearing
benchmark. Static process settings are classified as context unless a future
adapter can prove they are time-aligned controller commands.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from forgeworld.data.contracts import ColumnRole, ColumnSpec, DatasetSchema
from forgeworld.data.leakage import LeakageAuditReport, run_leakage_audit
from forgeworld.data.splits import SplitAssignment, SplitManifest
from forgeworld.data.timestamps import TimestampAudit, validate_numeric_sequence


LEGACY_CONTEXT_COLUMNS = frozenset(
    {
        "MillingToolType",
        "ADOC",
        "RDOC",
        "HardnessMean",
        "ToolHolderLength",
        "ToolRotation",
        "FeedRate",
        "ToolDiameter",
    }
)
LEGACY_FORBIDDEN_COLUMNS = frozenset({"CycleToFailure", "CycleToFailureNormalized"})
LEGACY_OUTCOME_COLUMNS = frozenset({"wear_class_name"})
LEGACY_ID_COLUMNS = frozenset({"FileName", "ToolIndex", "split"})


def default_legacy_cnc_manifest_path() -> Path:
    """Return the conventional sibling-repo manifest location."""

    return Path.cwd().parent / "industrial_jepa_mvp" / "data" / "manifests" / "cnc_windows.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "val":
        return "validation"
    if normalized in {"train", "validation", "test"}:
        return normalized
    raise ValueError(f"Unknown split value: {value!r}")


@dataclass(frozen=True)
class LegacyCncWindowsAdapter:
    manifest_csv: Path
    dataset_id: str = "legacy_mvp_cnc_windows"
    version: str = "industrial_jepa_mvp_54bf4099"

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_csv", Path(self.manifest_csv))

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Missing legacy CNC manifest: {self.manifest_csv}")
        with self.manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def build_manifest(self) -> Mapping[str, Any]:
        rows = self._read_rows()
        columns = list(rows[0]) if rows else []
        split_counts = Counter(_canonical_split(row["split"]) for row in rows if row.get("split"))
        tool_count = len({row.get("ToolIndex", "") for row in rows})
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source_file": str(self.manifest_csv),
            "source_sha256": _sha256(self.manifest_csv),
            "records": len(rows),
            "columns": columns,
            "tool_groups": tool_count,
            "split_counts": dict(sorted(split_counts.items())),
            "raw_data_committed": False,
            "claim_scope": "legacy_manifest_audit_only",
        }

    def validate_timestamps(self) -> TimestampAudit:
        rows = self._read_rows()
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                str(row.get("ToolIndex", "")),
                float(row.get("NumberOfCycle", "nan")),
                str(row.get("FileName", "")),
            ),
        )
        return validate_numeric_sequence(
            sorted_rows,
            "NumberOfCycle",
            ("ToolIndex",),
            require_strictly_increasing=False,
        )

    def classify_columns(self) -> DatasetSchema:
        rows = self._read_rows()
        columns = list(rows[0]) if rows else []
        specs: list[ColumnSpec] = []
        for column in columns:
            if column == "NumberOfCycle":
                specs.append(
                    ColumnSpec(
                        column,
                        ColumnRole.TIMESTAMP,
                        "integer",
                        unit="cycle",
                        description="Cycle/order index, not a wall-clock timestamp.",
                        allowed_as_model_input=False,
                        metadata={"temporal_kind": "cycle_index"},
                    )
                )
            elif column in LEGACY_ID_COLUMNS:
                specs.append(ColumnSpec(column, ColumnRole.ID, "string", allowed_as_model_input=False))
            elif column in LEGACY_CONTEXT_COLUMNS:
                specs.append(
                    ColumnSpec(
                        column,
                        ColumnRole.CONTEXT,
                        "float",
                        description=(
                            "Static process or asset context in the legacy manifest; "
                            "not verified as a time-aligned controller action."
                        ),
                    )
                )
            elif column in LEGACY_FORBIDDEN_COLUMNS:
                specs.append(
                    ColumnSpec(
                        column,
                        ColumnRole.FORBIDDEN,
                        "float",
                        description="Future lifecycle label prohibited as model input.",
                        allowed_as_model_input=False,
                    )
                )
            elif column in LEGACY_OUTCOME_COLUMNS:
                specs.append(
                    ColumnSpec(
                        column,
                        ColumnRole.OUTCOME,
                        "string",
                        description="Wear class derived from future lifecycle labels.",
                        allowed_as_model_input=False,
                    )
                )
            else:
                specs.append(
                    ColumnSpec(
                        column,
                        ColumnRole.UNKNOWN,
                        "string",
                        allowed_as_model_input=False,
                        description="Unmapped legacy column.",
                    )
                )
        return DatasetSchema(
            dataset_id=self.dataset_id,
            version=self.version,
            columns=tuple(specs),
            time_column="NumberOfCycle",
            id_columns=("FileName", "ToolIndex", "split"),
            group_columns=("ToolIndex",),
            outcome_columns=tuple(sorted(LEGACY_OUTCOME_COLUMNS & set(columns))),
            forbidden_columns=tuple(sorted(LEGACY_FORBIDDEN_COLUMNS & set(columns))),
            source=str(self.manifest_csv),
            metadata={
                "verified_action_columns": [],
                "legacy_columns_not_treated_as_actions": sorted(LEGACY_CONTEXT_COLUMNS & set(columns)),
            },
        )

    def build_grouped_splits(self) -> SplitManifest:
        rows = self._read_rows()
        group_to_split: dict[str, str] = {}
        assignments: list[SplitAssignment] = []
        for index, row in enumerate(rows):
            group = str(row["ToolIndex"])
            split = _canonical_split(row["split"])
            previous = group_to_split.setdefault(group, split)
            if previous != split:
                split = f"CONFLICT:{previous}:{split}"
            assignments.append(SplitAssignment(record_index=index, split=split, group=group))
        return SplitManifest(
            split_protocol="legacy_held_out_tool_manifest",
            group_key="ToolIndex",
            assignments=tuple(assignments),
            group_to_split=group_to_split,
        )

    def run_leakage_audit(self) -> LeakageAuditReport:
        schema = self.classify_columns()
        model_inputs = tuple(
            column.name
            for column in schema.columns
            if column.role in {ColumnRole.CONTEXT, ColumnRole.OBSERVATION}
            and column.effective_allowed_as_model_input
        )
        return run_leakage_audit(schema, model_inputs, self.build_grouped_splits())

    def build_audit_report(self) -> dict[str, Any]:
        schema = self.classify_columns()
        timestamp_audit = self.validate_timestamps()
        split_manifest = self.build_grouped_splits()
        leakage_report = self.run_leakage_audit()
        return {
            "manifest": self.build_manifest(),
            "schema": schema.to_dict(),
            "timestamp_audit": timestamp_audit.to_dict(),
            "split_manifest": split_manifest.to_dict(),
            "leakage_audit": leakage_report.to_dict(),
            "limitations": [
                "NumberOfCycle is an order index, not a wall-clock timestamp.",
                "Legacy static process settings are context, not verified actions.",
                "CycleToFailure and CycleToFailureNormalized are forbidden as inputs.",
                "This report audits a manifest; it does not reproduce model metrics.",
            ],
        }

    def write_audit_report(self, path: Path) -> Path:
        report = self.build_audit_report()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return path
