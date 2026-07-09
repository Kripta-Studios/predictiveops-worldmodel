"""Adapter for the preserved Industrial JEPA MVP MVTec AD bottle manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forgeworld.data.contracts import ColumnRole, ColumnSpec, DatasetSchema
from forgeworld.data.leakage import LeakageAuditReport, run_leakage_audit
from forgeworld.data.splits import SplitAssignment, SplitManifest
from forgeworld.data.timestamps import TimestampAudit, validate_numeric_sequence


def default_legacy_mvtec_data_root() -> Path:
    """Return the conventional sibling-repo data root for the legacy MVP."""

    return Path.cwd().parent / "industrial_jepa_mvp" / "data"


def default_legacy_mvtec_bottle_manifest_path() -> Path:
    return default_legacy_mvtec_data_root() / "manifests" / "mvtec_bottle.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"train", "validation", "test"}:
        return normalized
    raise ValueError(f"Unknown MVTec split value: {value!r}")


def _resolve_manifest_path(value: str, data_root: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0].lower() == "data":
        return data_root.parent / candidate
    return data_root / candidate


@dataclass(frozen=True)
class LegacyMvtecBottleAdapter:
    manifest_csv: Path
    data_root: Path
    validation_fraction: float = 0.2
    dataset_id: str = "legacy_mvp_mvtec_ad_bottle"
    version: str = "industrial_jepa_mvp_54bf4099"

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_csv", Path(self.manifest_csv))
        object.__setattr__(self, "data_root", Path(self.data_root))
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0).")

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Missing legacy MVTec manifest: {self.manifest_csv}")
        with self.manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for index, row in enumerate(rows):
            row["record_index"] = str(index)
        return rows

    def _protocol_split_by_index(self, rows: list[Mapping[str, str]]) -> dict[int, str]:
        train_indices = [
            index for index, row in enumerate(rows) if _canonical_split(row["split"]) == "train"
        ]
        validation_count = 0
        if len(train_indices) >= 2 and self.validation_fraction > 0:
            validation_count = max(1, round(len(train_indices) * self.validation_fraction))
            validation_count = min(validation_count, len(train_indices) - 1)
        validation_indices = set(train_indices[-validation_count:]) if validation_count else set()
        split_by_index: dict[int, str] = {}
        for index, row in enumerate(rows):
            original_split = _canonical_split(row["split"])
            if original_split == "train" and index in validation_indices:
                split_by_index[index] = "validation"
            else:
                split_by_index[index] = original_split
        return split_by_index

    def read_records(self) -> list[dict[str, Any]]:
        rows = self._read_rows()
        split_by_index = self._protocol_split_by_index(rows)
        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            image_path = _resolve_manifest_path(row.get("image", ""), self.data_root)
            mask_path = _resolve_manifest_path(row.get("mask", ""), self.data_root)
            record = {
                **row,
                "record_index": index,
                "label": int(row["label"]),
                "protocol_split": split_by_index[index],
                "image_path": str(image_path) if image_path is not None else "",
                "mask_path": str(mask_path) if mask_path is not None else "",
            }
            records.append(record)
        return records

    def build_manifest(self) -> Mapping[str, Any]:
        records = self.read_records()
        split_counts = Counter(record["protocol_split"] for record in records)
        label_counts = Counter(str(record["label"]) for record in records)
        defect_counts = Counter(str(record["defect_type"]) for record in records)
        missing_images = sum(1 for record in records if not Path(record["image_path"]).exists())
        missing_masks = sum(
            1
            for record in records
            if record["label"] == 1
            and record["mask_path"]
            and not Path(record["mask_path"]).exists()
        )
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source_file": str(self.manifest_csv),
            "source_sha256": _sha256(self.manifest_csv),
            "records": len(records),
            "split_counts": dict(sorted(split_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "defect_counts": dict(sorted(defect_counts.items())),
            "missing_images": missing_images,
            "missing_anomaly_masks": missing_masks,
            "raw_data_committed": False,
            "claim_scope": "legacy_visual_manifest_audit_only",
        }

    def validate_timestamps(self) -> TimestampAudit:
        return validate_numeric_sequence(
            self.read_records(),
            "record_index",
            ("category",),
            require_strictly_increasing=True,
        )

    def classify_columns(self) -> DatasetSchema:
        columns = (
            ColumnSpec(
                "record_index",
                ColumnRole.TIMESTAMP,
                "integer",
                description="Manifest row order; no acquisition timestamp is available.",
                allowed_as_model_input=False,
                metadata={"temporal_kind": "manifest_order"},
            ),
            ColumnSpec("split", ColumnRole.ID, "string", allowed_as_model_input=False),
            ColumnSpec(
                "protocol_split",
                ColumnRole.ID,
                "string",
                allowed_as_model_input=False,
                description="Official train/test plus train-normal validation split.",
            ),
            ColumnSpec("category", ColumnRole.CONTEXT, "string"),
            ColumnSpec(
                "image",
                ColumnRole.OBSERVATION,
                "path",
                unit="image_path",
                description="Relative RGB image path.",
            ),
            ColumnSpec(
                "image_path",
                ColumnRole.OBSERVATION,
                "path",
                unit="image_path",
                description="Resolved RGB image path.",
            ),
            ColumnSpec(
                "label",
                ColumnRole.OUTCOME,
                "integer",
                description="Image-level anomaly label; prohibited as input.",
                allowed_as_model_input=False,
            ),
            ColumnSpec(
                "defect_type",
                ColumnRole.OUTCOME,
                "string",
                description="Defect class; prohibited as input.",
                allowed_as_model_input=False,
            ),
            ColumnSpec(
                "mask",
                ColumnRole.FORBIDDEN,
                "path",
                description="Pixel mask must never be used by image-level baselines.",
                allowed_as_model_input=False,
            ),
            ColumnSpec(
                "mask_path",
                ColumnRole.FORBIDDEN,
                "path",
                description="Resolved pixel mask path; prohibited as input.",
                allowed_as_model_input=False,
            ),
        )
        return DatasetSchema(
            dataset_id=self.dataset_id,
            version=self.version,
            columns=columns,
            time_column="record_index",
            id_columns=("split", "protocol_split"),
            group_columns=("image_path",),
            outcome_columns=("label", "defect_type"),
            forbidden_columns=("mask", "mask_path"),
            source=str(self.manifest_csv),
            metadata={
                "verified_action_columns": [],
                "not_action_dataset": True,
                "validation_policy": "last_fraction_of_official_train_good_by_manifest_order",
            },
        )

    def build_grouped_splits(self) -> SplitManifest:
        records = self.read_records()
        group_to_split = {
            str(record["image_path"]): str(record["protocol_split"]) for record in records
        }
        assignments = tuple(
            SplitAssignment(
                record_index=int(record["record_index"]),
                split=str(record["protocol_split"]),
                group=str(record["image_path"]),
            )
            for record in records
        )
        return SplitManifest(
            split_protocol="official_train_test_plus_train_normal_validation",
            group_key="image_path",
            assignments=assignments,
            group_to_split=group_to_split,
        )

    def run_leakage_audit(self) -> LeakageAuditReport:
        schema = self.classify_columns()
        model_inputs = ("category", "image_path")
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
                "MVTec AD is a visual anomaly dataset, not an action-conditioned process dataset.",
                "The manifest has no acquisition timestamps; record_index is only manifest order.",
                "Pixel masks are forbidden for image-level score training and threshold selection.",
                "The validation split is carved from official train-good images only.",
            ],
        }

    def write_audit_report(self, path: Path) -> Path:
        report = self.build_audit_report()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return path
