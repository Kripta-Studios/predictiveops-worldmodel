"""Dataset adapter protocol required by AGENTS.md."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from forgeworld.data.contracts import DatasetSchema
from forgeworld.data.leakage import LeakageAuditReport
from forgeworld.data.splits import SplitManifest
from forgeworld.data.timestamps import TimestampAudit


class DatasetAdapter(Protocol):
    """Required interface for every dataset adapter."""

    def build_manifest(self) -> Mapping[str, Any]:
        """Return immutable source, version, checksum, and extraction metadata."""

    def validate_timestamps(self) -> TimestampAudit:
        """Validate time ordering and parseability before any split/model use."""

    def classify_columns(self) -> DatasetSchema:
        """Classify every column as action, context, observation, outcome, id, or forbidden."""

    def build_grouped_splits(self) -> SplitManifest:
        """Build non-row-random grouped splits appropriate for lifecycle/process data."""

    def run_leakage_audit(self) -> LeakageAuditReport:
        """Audit forbidden columns, timestamp leakage, and split isolation."""
