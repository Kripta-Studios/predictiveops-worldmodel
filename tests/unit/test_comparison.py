from __future__ import annotations

from forgeworld.evaluation.comparison import compare_world_model_to_baselines


def test_comparison_records_when_simple_baseline_has_higher_metrics() -> None:
    world = {
        "experiment_name": "wm",
        "dataset": "d",
        "dataset_version": "v",
        "split_protocol": "grouped",
        "seed_count": 3,
        "claim_scope": "experimental",
        "test_summary": {
            "failure_auroc": {"mean": 0.8},
            "failure_auprc": {"mean": 0.5},
            "forecast_mse": {"mean": 1.2},
        },
    }
    baseline = {
        "experiment_name": "baseline",
        "seeds": [0, 1, 2],
        "summary": [
            {"model": "majority", "feature_set": "sensor", "auroc_mean": 0.5, "auprc_mean": 0.1},
            {"model": "tree", "feature_set": "sensor", "auroc_mean": 0.9, "auprc_mean": 0.7},
        ],
    }

    report = compare_world_model_to_baselines(world, baseline)

    assert report["outcome"] == "simple_baseline_has_higher_recorded_metrics_on_this_protocol"
    assert report["deltas"]["auroc_vs_highest_baseline"] < 0
