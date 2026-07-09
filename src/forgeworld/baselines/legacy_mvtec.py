"""Simple MVTec AD bottle visual baselines with leakage-guarded thresholding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forgeworld.data.datasets.legacy_mvtec import (
    LegacyMvtecBottleAdapter,
    default_legacy_mvtec_bottle_manifest_path,
    default_legacy_mvtec_data_root,
)
from forgeworld.runtime.compute import WorkerPlan


def _require_visual_dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        from PIL import Image
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent.
        raise RuntimeError(
            "Visual baselines require numpy, pillow, and scikit-learn. "
            "Install with the visual-baselines optional dependency group."
        ) from exc
    return (
        np,
        Image,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@dataclass(frozen=True)
class LegacyMvtecBaselineConfig:
    manifest_csv: Path = default_legacy_mvtec_bottle_manifest_path()
    data_root: Path = default_legacy_mvtec_data_root()
    output_dir: Path = Path("outputs/baselines/legacy_mvtec_bottle_simple")
    category: str = "bottle"
    validation_fraction: float = 0.2
    validation_percentile: float = 95.0
    image_size: int = 32
    feature_sets: tuple[str, ...] = ("color_stats", "thumbnail_gray")
    claim_scope: str = "visual_baseline_smoke_not_claim_bearing"
    unsafe_debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_csv", Path(self.manifest_csv))
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def _image_array(path: Path, image_size: int) -> Any:
    np, Image, *_metrics = _require_visual_dependencies()
    resample = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), resample)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _features_for_image(path: Path, feature_set: str, image_size: int) -> Any:
    np = _require_visual_dependencies()[0]
    array = _image_array(path, image_size)
    if feature_set == "color_stats":
        axes = (0, 1)
        return np.concatenate(
            [
                array.mean(axis=axes),
                array.std(axis=axes),
                array.min(axis=axes),
                array.max(axis=axes),
            ]
        ).astype(np.float32)
    if feature_set == "thumbnail_gray":
        gray = array.mean(axis=2)
        return gray.reshape(-1).astype(np.float32)
    raise ValueError(f"Unknown MVTec feature set: {feature_set}")


def _feature_matrix(records: list[dict[str, Any]], feature_set: str, image_size: int) -> Any:
    np = _require_visual_dependencies()[0]
    features = [
        _features_for_image(Path(record["image_path"]), feature_set, image_size)
        for record in records
    ]
    return np.vstack(features).astype(np.float32)


def _standardize(train: Any, *others: Any) -> tuple[Any, ...]:
    np = _require_visual_dependencies()[0]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return tuple((matrix - mean) / std for matrix in (train, *others))


def _nearest_train_distance(train: Any, query: Any, chunk_size: int = 64) -> Any:
    np = _require_visual_dependencies()[0]
    scores: list[Any] = []
    train_sq = np.sum(train * train, axis=1, keepdims=True).T
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances = np.maximum(chunk_sq + train_sq - 2.0 * chunk @ train.T, 0.0)
        scores.append(np.sqrt(distances.min(axis=1)))
    return np.concatenate(scores)


def _metric_payload(labels: Any, scores: Any, alerts: Any) -> dict[str, float | None]:
    (
        np,
        _Image,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    ) = _require_visual_dependencies()
    unique = set(int(value) for value in labels.tolist())
    auroc = float(roc_auc_score(labels, scores)) if len(unique) > 1 else None
    auprc = float(average_precision_score(labels, scores)) if len(unique) > 1 else None
    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": float(precision_score(labels, alerts, zero_division=0)),
        "recall": float(recall_score(labels, alerts, zero_division=0)),
        "f1": float(f1_score(labels, alerts, zero_division=0)),
        "alert_rate": float(np.mean(alerts)) if len(alerts) else None,
    }


def _defect_breakdown(
    records: list[dict[str, Any]], scores: Any, alerts: Any
) -> list[dict[str, Any]]:
    np = _require_visual_dependencies()[0]
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record["defect_type"]), []).append(index)
    breakdown: list[dict[str, Any]] = []
    for defect_type, indices in sorted(grouped.items()):
        labels = np.asarray([records[index]["label"] for index in indices], dtype=np.int64)
        defect_alerts = alerts[indices]
        breakdown.append(
            {
                "defect_type": defect_type,
                "count": len(indices),
                "positive_count": int(labels.sum()),
                "mean_score": float(np.mean(scores[indices])),
                "recall": (
                    float(np.mean(defect_alerts[labels == 1])) if int(labels.sum()) else None
                ),
                "false_positive_rate": (
                    float(np.mean(defect_alerts[labels == 0]))
                    if int((labels == 0).sum())
                    else None
                ),
            }
        )
    return breakdown


def run_legacy_mvtec_baselines(
    config: LegacyMvtecBaselineConfig,
    worker_plan: WorkerPlan,
) -> dict[str, Any]:
    """Run simple visual baselines without using test labels for threshold selection."""

    np = _require_visual_dependencies()[0]
    adapter = LegacyMvtecBottleAdapter(
        manifest_csv=config.manifest_csv,
        data_root=config.data_root,
        validation_fraction=config.validation_fraction,
    )
    leakage = adapter.run_leakage_audit().assert_training_allowed(config.unsafe_debug)
    records = [record for record in adapter.read_records() if record["category"] == config.category]
    train_records = [record for record in records if record["protocol_split"] == "train"]
    validation_records = [
        record for record in records if record["protocol_split"] == "validation"
    ]
    test_records = [record for record in records if record["protocol_split"] == "test"]
    if not train_records or not validation_records or not test_records:
        counts = {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        }
        raise RuntimeError(f"Empty MVTec split after applying protocol: {counts}")
    if any(int(record["label"]) != 0 for record in train_records + validation_records):
        raise RuntimeError("MVTec train/validation records must be normal-only for this baseline.")

    rows: list[dict[str, Any]] = []
    for feature_set in config.feature_sets:
        x_train = _feature_matrix(train_records, feature_set, config.image_size)
        x_validation = _feature_matrix(validation_records, feature_set, config.image_size)
        x_test = _feature_matrix(test_records, feature_set, config.image_size)
        x_train, x_validation, x_test = _standardize(x_train, x_validation, x_test)
        validation_scores = _nearest_train_distance(x_train, x_validation)
        test_scores = _nearest_train_distance(x_train, x_test)
        threshold = float(np.percentile(validation_scores, config.validation_percentile))
        labels = np.asarray([record["label"] for record in test_records], dtype=np.int64)
        alerts = (test_scores > threshold).astype(np.int64)
        rows.append(
            {
                "model": "nearest_train_feature_distance",
                "feature_set": feature_set,
                "feature_count": int(x_train.shape[1]),
                "threshold": threshold,
                "threshold_selection": (
                    f"validation_normal_score_percentile_{config.validation_percentile:g}"
                ),
                "validation_score_count": int(len(validation_scores)),
                "test_score_count": int(len(test_scores)),
                "defect_breakdown": _defect_breakdown(test_records, test_scores, alerts),
                **_metric_payload(labels, test_scores, alerts),
            }
        )

    split_manifest = adapter.build_grouped_splits().to_dict()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = config.output_dir / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    split_counts = {
        "train": len(train_records),
        "validation": len(validation_records),
        "test": len(test_records),
    }
    report = {
        "experiment_name": "legacy_mvtec_bottle_simple_visual_baselines",
        "hypothesis": (
            "A leakage-guarded nearest-neighbor baseline on simple image features "
            "sets a visual sanity floor before PatchCore, PaDiM, EfficientAD, or "
            "foundation-feature baselines are added."
        ),
        "dataset": adapter.dataset_id,
        "dataset_version": adapter.version,
        "category": config.category,
        "split_protocol": split_manifest["split_protocol"],
        "split_counts": split_counts,
        "threshold_selection": "validation_normal_scores_only_no_test_label_tuning",
        "metrics": ["auroc", "auprc", "precision", "recall", "f1", "alert_rate"],
        "claim_scope": config.claim_scope,
        "test_set_influenced_model_selection": False,
        "feature_sets": list(config.feature_sets),
        "worker_plan": worker_plan.to_dict(),
        "leakage_audit": leakage.to_dict(),
        "split_manifest": {
            key: split_manifest[key]
            for key in ("split_protocol", "group_key", "counts", "groups_by_split")
        },
        "split_manifest_path": str(split_manifest_path),
        "runs": rows,
        "what_this_result_does_not_prove": [
            "It does not prove visual benchmark competitiveness.",
            "It does not replace PatchCore, PaDiM, EfficientAD, or frozen foundation features.",
            "It does not prove action-conditioned or process-world modeling.",
            "It does not tune thresholds with anomalous validation labels.",
        ],
        "limitations": [
            "This is an image-level smoke baseline on one MVTec AD category.",
            "MVTec AD contains visual defects, not machine commands or process actions.",
            "Pixel masks are not used; pixel-level AU-PRO/AUPRC is not reported.",
            "The validation split is carved from official train-good images only.",
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
                "manifest_csv": str(config.manifest_csv),
                "data_root": str(config.data_root),
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report
