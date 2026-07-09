"""Interfaces for modules that may be called world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class LatentState:
    values: Sequence[Sequence[float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictiveDistribution:
    horizon: int
    mean: Sequence[Sequence[float]]
    uncertainty: Sequence[Sequence[float]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurpriseBreakdown:
    total: float
    by_signal: Mapping[str, float] = field(default_factory=dict)
    by_component: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WorldModel(Protocol):
    """Required interface for a future-state predictor."""

    def encode(self, observation_history: Any) -> LatentState:
        """Encode historical observations into a latent state."""

    def predict(
        self,
        latent_state: LatentState,
        actions: Any,
        context: Any,
        horizon: int,
    ) -> PredictiveDistribution:
        """Predict a future distribution conditioned on history, actions, and context."""

    def score(self, prediction: PredictiveDistribution, observation: Any) -> SurpriseBreakdown:
        """Score observed data against the predictive distribution."""
