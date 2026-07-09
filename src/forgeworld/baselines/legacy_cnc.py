"""Simple CNC failure-soon baselines with grouped split and leakage guard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from forgeworld.data.datasets.legacy_cnc import LegacyCncWindowsAdapter
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
FORBIDDEN_COLUMNS = {"CycleToFailure", "CycleToFailureNormalized", "wear_class", "wear_class_name"}


@dataclass(frozen=True)
class LegacyCncBaselineConfig:
    raw_feature_csv: Path
    manifest_csv: Path
    output_dir: Path
    failure_horizon_cycles: int = 10
    seeds: tuple[int, ...] = (0, 1, 2)
    model_names: tuple[str, ...] = ("majority", "logistic_regression", "random_forest")
    feature_sets: tuple[str, ...] = ("context_only", "sensor_only", "sensor_plus_context")
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


def _split_masks_from_manifest(df: Any, manifest_csv: Path) -> dict[str, Any]:
    adapter = LegacyCncWindowsAdapter(manifest_csv)
    split_manifest = adapter.build_grouped_splits()
    group_to_split = split_manifest.group_to_split
    tool_split = df["ToolIndex"].astype(str).map(group_to_split)
    return {
        "train": tool_split == "train",
        "validation": tool_split == "validation",
        "test": tool_split == "test",
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
        return make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="most_frequent"))
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


def _aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["feature_set"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model, feature_set), values in sorted(grouped.items()):
        summary: dict[str, Any] = {"model": model, "feature_set": feature_set, "seeds": len(values)}
        for metric in ("accuracy", "f1", "auroc", "auprc"):
            present = [value[metric] for value in values if value[metric] is not None]
            summary[f"{metric}_mean"] = float(mean(present)) if present else None
            summary[f"{metric}_std"] = float(stdev(present)) if len(present) > 1 else 0.0 if present else None
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
    masks = _split_masks_from_manifest(df, config.manifest_csv)
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
        x_test = df.loc[masks["test"], columns].to_numpy(dtype=np.float32)
        y_test = df.loc[masks["test"], "failure_soon"].to_numpy(dtype=np.int64)
        for seed in config.seeds:
            for model_name in config.model_names:
                model = _build_model(model_name, seed, worker_plan.baseline_parallel_jobs)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_score = _positive_scores(model, x_test)
                metrics = _metric_payload(y_test, y_pred, y_score)
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "seed": seed,
                        "feature_count": len(columns),
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
    report = {
        "experiment_name": "legacy_cnc_failure_soon_simple_baselines",
        "hypothesis": (
            "Simple context, engineered-sensor, and tree/linear baselines establish "
            "a leakage-guarded floor before neural models."
        ),
        "dataset": adapter.dataset_id,
        "dataset_version": adapter.version,
        "split_protocol": "legacy_held_out_tool_manifest",
        "seeds": list(config.seeds),
        "failure_horizon_cycles": config.failure_horizon_cycles,
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "threshold_selection": "fixed_0_5_no_test_tuning",
        "claim_scope": config.claim_scope,
        "test_set_influenced_model_selection": False,
        "split_counts": split_counts,
        "feature_columns_by_set": feature_columns_by_set,
        "worker_plan": worker_plan.to_dict(),
        "leakage_audit": leakage.to_dict(),
        "runs": rows,
        "summary": summary,
        "warnings": warnings,
        "limitations": [
            "This is a row-level smoke baseline on engineered feature rows, not a full JEPA run.",
            "The legacy CNC manifest provides context only; no verified action claims are made.",
            "No threshold was tuned on validation or test labels.",
            "Perfect smoke metrics must be treated as a leakage/protocol review trigger.",
        ],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
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
