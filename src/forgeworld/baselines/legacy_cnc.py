"""Simple CNC failure-soon baselines with grouped split and leakage guard."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from forgeworld.data.datasets.legacy_cnc import LegacyCncWindowsAdapter
from forgeworld.evaluation.lead_time import event_lead_time_metrics
from forgeworld.runtime.compute import WorkerPlan

META_COLUMNS = {
    "FileName",
    "NumberOfCycle",
    "SampleIndex",
    "TollIndex",
    "ToolIndex",
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength",
    "ToolRotation",
    "FeedRate",
    "ToolDiameter",
    "CycleToFailure",
    "CycleToFailureNormalized",
    "wear_class",
    "wear_class_name",
}
CONTEXT_COLUMNS = (
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength",
    "ToolRotation",
    "FeedRate",
    "ToolDiameter",
)
FORBIDDEN_COLUMNS = {
    "CycleToFailure",
    "CycleToFailureNormalized",
    "failure_soon",
    "wear_class",
    "wear_class_name",
}


@dataclass(frozen=True)
class LegacyCncBaselineConfig:
    raw_feature_csv: Path
    manifest_csv: Path
    output_dir: Path
    failure_horizon_cycles: int = 10
    seeds: tuple[int, ...] = (0, 1, 2)
    model_names: tuple[str, ...] = ("majority", "logistic_regression", "random_forest")
    feature_sets: tuple[str, ...] = ("context_only", "sensor_only", "sensor_plus_context")
    split_protocol: str = "legacy_held_out_tool_manifest"
    claim_scope: str = "baseline_smoke_not_claim_bearing"
    unsafe_debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_feature_csv", Path(self.raw_feature_csv))
        object.__setattr__(self, "manifest_csv", Path(self.manifest_csv))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def _require_baseline_dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Baseline dependencies are missing. Install with `pip install -e .[baselines]`."
        ) from exc
    return (
        np,
        pd,
        DummyClassifier,
        RandomForestClassifier,
        SimpleImputer,
        LogisticRegression,
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
        make_pipeline,
        StandardScaler,
    )


def load_legacy_cnc_feature_table(raw_feature_csv: Path) -> Any:
    """Load the legacy feature table without using lifecycle labels as inputs."""

    (
        _np,
        pd,
        *_rest,
    ) = _require_baseline_dependencies()
    if not raw_feature_csv.exists():
        raise FileNotFoundError(f"Missing CNC feature table: {raw_feature_csv}")
    df = pd.read_csv(raw_feature_csv, sep=";", header=1, decimal=",", low_memory=False)
    if "TollIndex" in df.columns and "ToolIndex" not in df.columns:
        df["ToolIndex"] = df["TollIndex"]
    numeric_candidates = [column for column in df.columns if column not in {"FileName"}]
    for column in numeric_candidates:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.replace(",", ".", regex=False)
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def select_feature_columns(df: Any, feature_set: str) -> list[str]:
    pd = _require_baseline_dependencies()[1]
    context = [
        column
        for column in CONTEXT_COLUMNS
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    sensor = [
        column
        for column in df.columns
        if column not in META_COLUMNS
        and column not in FORBIDDEN_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if feature_set == "context_only":
        selected = context
    elif feature_set == "sensor_only":
        selected = sensor
    elif feature_set == "sensor_plus_context":
        selected = sensor + context
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    forbidden_used = sorted(FORBIDDEN_COLUMNS & set(selected))
    if forbidden_used:
        raise RuntimeError(f"Forbidden columns selected as features: {forbidden_used}")
    if not selected:
        raise RuntimeError(f"No columns available for feature set {feature_set!r}.")
    return selected


def _split_masks_from_manifest(
    df: Any, manifest_csv: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter = LegacyCncWindowsAdapter(manifest_csv)
    split_manifest = adapter.build_grouped_splits()
    group_to_split = split_manifest.group_to_split
    tool_split = df["ToolIndex"].astype(str).map(group_to_split)
    masks = {
        "train": tool_split == "train",
        "validation": tool_split == "validation",
        "test": tool_split == "test",
    }
    return masks, split_manifest.to_dict()


def _split_masks_from_cutting_condition(df: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if "ADOC" not in df.columns or "RDOC" not in df.columns:
        raise RuntimeError("held_out_cutting_condition split requires ADOC and RDOC columns.")
    groups = (df["ADOC"].astype(str) + "_rdoc_" + df["RDOC"].astype(str)).rename(
        "cutting_condition"
    )
    unique_groups = sorted(groups.dropna().unique().tolist())
    if len(unique_groups) < 3:
        raise RuntimeError(
            f"held_out_cutting_condition requires at least 3 groups, found {len(unique_groups)}."
        )
    # Deterministic hardest split: train on early sorted groups, validate/test on held-out groups.
    group_to_split = {
        group: (
            "train"
            if index < len(unique_groups) - 2
            else "validation"
            if index == len(unique_groups) - 2
            else "test"
        )
        for index, group in enumerate(unique_groups)
    }
    split = groups.map(group_to_split)
    masks = {
        "train": split == "train",
        "validation": split == "validation",
        "test": split == "test",
    }
    assignments = [
        {"record_index": int(index), "split": str(group_to_split[group]), "group": str(group)}
        for index, group in enumerate(groups)
    ]
    manifest = {
        "split_protocol": "held_out_cutting_condition",
        "group_key": "ADOC_RDOC",
        "groups_by_split": {
            split_name: tuple(
                group for group in unique_groups if group_to_split[group] == split_name
            )
            for split_name in ("train", "validation", "test")
        },
        "counts": {split_name: int(mask.sum()) for split_name, mask in masks.items()},
        "assignments": assignments,
    }
    return masks, manifest


def _build_split_masks(
    df: Any,
    config: LegacyCncBaselineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.split_protocol == "legacy_held_out_tool_manifest":
        return _split_masks_from_manifest(df, config.manifest_csv)
    if config.split_protocol == "held_out_cutting_condition":
        return _split_masks_from_cutting_condition(df)
    raise ValueError(f"Unknown split protocol: {config.split_protocol}")


def _compact_split_manifest(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split_protocol": split_manifest["split_protocol"],
        "group_key": split_manifest["group_key"],
        "counts": split_manifest["counts"],
        "groups_by_split": split_manifest["groups_by_split"],
    }


def _build_model(model_name: str, seed: int, n_jobs: int) -> Any:
    (
        _np,
        _pd,
        DummyClassifier,
        RandomForestClassifier,
        SimpleImputer,
        LogisticRegression,
        *_metrics_and_pipeline,
        make_pipeline,
        StandardScaler,
    ) = _require_baseline_dependencies()
    if model_name == "majority":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DummyClassifier(strategy="most_frequent"),
        )
    if model_name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    if model_name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=n_jobs,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _positive_scores(model: Any, x: Any) -> Any:
    np = _require_baseline_dependencies()[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.shape[1] == 1:
            return np.zeros(probabilities.shape[0], dtype=float)
        return probabilities[:, 1]
    return model.predict(x)


def _metric_payload(y_true: Any, y_pred: Any, y_score: Any) -> dict[str, float | None]:
    (
        _np,
        _pd,
        _DummyClassifier,
        _RandomForestClassifier,
        _SimpleImputer,
        _LogisticRegression,
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
        *_rest,
    ) = _require_baseline_dependencies()
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(int(value) for value in y_true)) < 2:
        metrics["auroc"] = None
        metrics["auprc"] = None
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    return metrics


def _operational_payload(df_split: Any, y_true: Any, y_pred: Any) -> dict[str, float | int | None]:
    records = [
        {
            "asset_id": str(tool),
            "cycle": float(cycle),
            "label": int(label),
            "alert": int(alert),
        }
        for tool, cycle, label, alert in zip(
            df_split["ToolIndex"],
            df_split["NumberOfCycle"],
            y_true,
            y_pred,
            strict=True,
        )
    ]
    return event_lead_time_metrics(records).to_dict()


def _aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["feature_set"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model, feature_set), values in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "feature_set": feature_set,
            "seeds": len(values),
        }
        for metric in ("accuracy", "f1", "auroc", "auprc"):
            present = [value[metric] for value in values if value[metric] is not None]
            summary[f"{metric}_mean"] = float(mean(present)) if present else None
            summary[f"{metric}_std"] = (
                float(stdev(present)) if len(present) > 1 else 0.0 if present else None
            )
        for metric in (
            "event_recall",
            "median_lead_cycles",
            "events_missed",
            "false_alarms_per_1000_cycles",
        ):
            present = [
                value["operational"][metric]
                for value in values
                if value.get("operational", {}).get(metric) is not None
            ]
            summary[f"{metric}_mean"] = float(mean(present)) if present else None
            summary[f"{metric}_std"] = (
                float(stdev(present)) if len(present) > 1 else 0.0 if present else None
            )
        summaries.append(summary)
    return summaries


def run_legacy_cnc_baselines(
    config: LegacyCncBaselineConfig,
    worker_plan: WorkerPlan,
) -> dict[str, Any]:
    """Run simple grouped-split baselines after leakage audit passes."""

    np = _require_baseline_dependencies()[0]
    adapter = LegacyCncWindowsAdapter(config.manifest_csv)
    leakage = adapter.run_leakage_audit().assert_training_allowed(config.unsafe_debug)
    df = load_legacy_cnc_feature_table(config.raw_feature_csv)
    df = df.dropna(subset=["ToolIndex", "CycleToFailure"]).copy()
    df["failure_soon"] = (df["CycleToFailure"] <= config.failure_horizon_cycles).astype(int)
    masks, split_manifest = _build_split_masks(df, config)
    split_counts = {split: int(mask.sum()) for split, mask in masks.items()}
    if min(split_counts.values()) <= 0:
        raise RuntimeError(f"Empty split after applying manifest group split: {split_counts}")

    rows: list[dict[str, Any]] = []
    feature_columns_by_set: dict[str, list[str]] = {}
    for feature_set in config.feature_sets:
        columns = select_feature_columns(df, feature_set)
        feature_columns_by_set[feature_set] = columns
        x_train = df.loc[masks["train"], columns].to_numpy(dtype=np.float32)
        y_train = df.loc[masks["train"], "failure_soon"].to_numpy(dtype=np.int64)
        test_df = df.loc[masks["test"], ["ToolIndex", "NumberOfCycle"]].copy()
        x_test = df.loc[masks["test"], columns].to_numpy(dtype=np.float32)
        y_test = df.loc[masks["test"], "failure_soon"].to_numpy(dtype=np.int64)
        for seed in config.seeds:
            for model_name in config.model_names:
                model = _build_model(model_name, seed, worker_plan.baseline_parallel_jobs)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_score = _positive_scores(model, x_test)
                metrics = _metric_payload(y_test, y_pred, y_score)
                operational = _operational_payload(test_df, y_test, y_pred)
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "seed": seed,
                        "feature_count": len(columns),
                        "operational": operational,
                        **metrics,
                    }
                )

    summary = _aggregate(rows)
    warnings = [
        (
            "perfect_test_metric_detected_manual_protocol_review_required:"
            f"{row['model']}:{row['feature_set']}:seed_{row['seed']}"
        )
        for row in rows
        if row["accuracy"] == 1.0 and row["auroc"] == 1.0
    ]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = config.output_dir / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = {
        "experiment_name": "legacy_cnc_failure_soon_simple_baselines",
        "hypothesis": (
            "Simple context, engineered-sensor, and tree/linear baselines establish "
            "a leakage-guarded floor before neural models."
        ),
        "dataset": adapter.dataset_id,
        "dataset_version": adapter.version,
        "split_protocol": config.split_protocol,
        "seeds": list(config.seeds),
        "failure_horizon_cycles": config.failure_horizon_cycles,
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "metrics": ["accuracy", "f1", "auroc", "auprc"],
        "operational_metrics": [
            "event_recall",
            "median_lead_cycles",
            "events_missed",
            "false_alarms_per_1000_cycles",
        ],
        "threshold_selection": "fixed_0_5_no_test_tuning",
        "claim_scope": config.claim_scope,
        "test_set_influenced_model_selection": False,
        "split_counts": split_counts,
        "split_manifest": _compact_split_manifest(split_manifest),
        "split_manifest_path": str(split_manifest_path),
        "feature_columns_by_set": feature_columns_by_set,
        "worker_plan": worker_plan.to_dict(),
        "leakage_audit": leakage.to_dict(),
        "runs": rows,
        "summary": summary,
        "warnings": warnings,
        "what_this_result_does_not_prove": [
            "It does not prove production readiness or external generalization.",
            (
                "It does not prove action-conditioned or causal modeling; "
                "legacy process settings are context."
            ),
            "It does not prove a neural world model adds value over engineered sensor features.",
            "It does not validate threshold selection beyond the fixed 0.5 smoke setting.",
        ],
        "limitations": [
            "This is a row-level smoke baseline on engineered feature rows, not a full JEPA run.",
            "The legacy CNC manifest provides context only; no verified action claims are made.",
            "No threshold was tuned on validation or test labels.",
            "Perfect smoke metrics must be treated as a leakage/protocol review trigger.",
        ],
    }
    (config.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (config.output_dir / "config_resolved.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "raw_feature_csv": str(config.raw_feature_csv),
                "manifest_csv": str(config.manifest_csv),
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report
