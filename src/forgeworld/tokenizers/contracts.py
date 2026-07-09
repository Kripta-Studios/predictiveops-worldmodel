"""Token batch contract shared by sensor, event, vision, audio, and quality tokenizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class TokenBatchError(ValueError):
    """Raised when a TokenBatch violates shape or metadata invariants."""


@dataclass(frozen=True)
class TokenBatch:
    values: Sequence[Sequence[float]]
    timestamps: Sequence[Any]
    semantic_ids: Sequence[str]
    asset_ids: Sequence[str]
    masks: Sequence[Sequence[bool]]
    units: Sequence[str | None]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def token_count(self) -> int:
        return len(self.values)

    @property
    def feature_count(self) -> int:
        return len(self.values[0]) if self.values else 0

    def validate(self) -> None:
        token_count = len(self.values)
        same_length_fields = {
            "timestamps": self.timestamps,
            "semantic_ids": self.semantic_ids,
            "asset_ids": self.asset_ids,
            "masks": self.masks,
            "units": self.units,
        }
        for field_name, field_value in same_length_fields.items():
            if len(field_value) != token_count:
                raise TokenBatchError(
                    f"{field_name} length {len(field_value)} does not match values length "
                    f"{token_count}."
                )
        if token_count == 0:
            return
        feature_count = len(self.values[0])
        for index, row in enumerate(self.values):
            if len(row) != feature_count:
                raise TokenBatchError(
                    f"values row {index} length {len(row)} does not match {feature_count}."
                )
        for index, mask in enumerate(self.masks):
            if len(mask) != feature_count:
                raise TokenBatchError(
                    f"mask row {index} length {len(mask)} does not match {feature_count}."
                )
