"""Compare experimental model reports against mandatory simple baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def compare_world_model_to_baselines(
    world_model_report: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable comparison without claim-bearing wording."""

    baseline_rows = baseline_report["summary"]
    world_test = world_model_report["test_summary"]

    def baseline_row_for(metric: str) -> dict[str, Any]:
        candidates = [row for row in baseline_rows if row.get(metric) is not None]
        if not candidates:
            raise ValueError(f"No baseline rows contain metric {metric!r}.")
        return max(candidates, key=lambda row: float(row[metric]))

    auroc_baseline = baseline_row_for("auroc_mean")
    auprc_baseline = baseline_row_for("auprc_mean")
    world_auroc = world_test["failure_auroc"]["mean"]
    world_auprc = world_test["failure_auprc"]["mean"]
    return {
        "comparison_name": "legacy_cnc_world_model_vs_simple_baselines",
        "dataset": world_model_report["dataset"],
        "dataset_version": world_model_report["dataset_version"],
        "split_protocol": world_model_report["split_protocol"],
        "world_model": {
            "experiment_name": world_model_report["experiment_name"],
            "seed_count": world_model_report["seed_count"],
            "failure_auroc_mean": world_auroc,
            "failure_auprc_mean": world_auprc,
            "forecast_mse_mean": world_test["forecast_mse"]["mean"],
            "test_operational_summary": world_model_report.get("test_operational_summary"),
            "claim_scope": world_model_report["claim_scope"],
        },
        "baseline_reference": {
            "experiment_name": baseline_report["experiment_name"],
            "seed_count": len(baseline_report["seeds"]),
            "highest_auroc_baseline": {
                "model": auroc_baseline["model"],
                "feature_set": auroc_baseline["feature_set"],
                "auroc_mean": auroc_baseline["auroc_mean"],
                "auprc_mean": auroc_baseline["auprc_mean"],
                "event_recall_mean": auroc_baseline.get("event_recall_mean"),
                "median_lead_cycles_mean": auroc_baseline.get("median_lead_cycles_mean"),
                "false_alarms_per_1000_cycles_mean": auroc_baseline.get(
                    "false_alarms_per_1000_cycles_mean"
                ),
            },
            "highest_auprc_baseline": {
                "model": auprc_baseline["model"],
                "feature_set": auprc_baseline["feature_set"],
                "auroc_mean": auprc_baseline["auroc_mean"],
                "auprc_mean": auprc_baseline["auprc_mean"],
                "event_recall_mean": auprc_baseline.get("event_recall_mean"),
                "median_lead_cycles_mean": auprc_baseline.get("median_lead_cycles_mean"),
                "false_alarms_per_1000_cycles_mean": auprc_baseline.get(
                    "false_alarms_per_1000_cycles_mean"
                ),
            },
        },
        "deltas": {
            "auroc_vs_highest_baseline": (
                None if world_auroc is None else world_auroc - auroc_baseline["auroc_mean"]
            ),
            "auprc_vs_highest_baseline": (
                None if world_auprc is None else world_auprc - auprc_baseline["auprc_mean"]
            ),
        },
        "outcome": (
            "simple_baseline_has_higher_recorded_metrics_on_this_protocol"
            if (
                world_auroc is None
                or world_auprc is None
                or world_auroc < auroc_baseline["auroc_mean"]
                or world_auprc < auprc_baseline["auprc_mean"]
            )
            else "world_model_has_higher_recorded_metrics_under_this_internal_protocol"
        ),
        "limitations": [
            "This is an internal protocol comparison, not an external benchmark claim.",
            "The CNC data has no verified action columns; this is context-conditioned.",
            "Perfect or near-perfect simple-baseline metrics require manual protocol review.",
        ],
        "next_falsifiable_hypothesis": (
            "The world model should add value on harder chronological or held-out-condition splits, "
            "or on lead-time/forecasting metrics not captured by row-level failure classification."
        ),
    }


def write_comparison_report(
    world_model_report_path: Path,
    baseline_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = compare_world_model_to_baselines(
        load_json(world_model_report_path),
        load_json(baseline_report_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
