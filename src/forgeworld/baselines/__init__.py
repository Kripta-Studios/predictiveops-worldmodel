"""Simple baseline runners required before neural model claims."""

from forgeworld.baselines.legacy_cnc import LegacyCncBaselineConfig, run_legacy_cnc_baselines
from forgeworld.baselines.legacy_mvtec import (
    LegacyMvtecBaselineConfig,
    run_legacy_mvtec_baselines,
)

__all__ = [
    "LegacyCncBaselineConfig",
    "LegacyMvtecBaselineConfig",
    "run_legacy_cnc_baselines",
    "run_legacy_mvtec_baselines",
]
