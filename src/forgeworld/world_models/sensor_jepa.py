"""Experimental deterministic multi-horizon Sensor-JEPA world model.

This module is intentionally scoped to the audited legacy CNC feature table. It
uses static process settings as context, not actions, because no time-aligned
controller commands are present in the manifest.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from forgeworld.baselines.legacy_cnc import (
    CONTEXT_COLUMNS,
    FORBIDDEN_COLUMNS,
    load_legacy_cnc_feature_table,
    select_feature_columns,
)
from forgeworld.data.datasets.legacy_cnc import LegacyCncWindowsAdapter
from forgeworld.runtime.compute import WorkerPlan


def _require_world_model_dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        import torch
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError(
            "World-model dependencies are missing. Install with `pip install -e .[world-models]`."
        ) from exc
    return np, torch, average_precision_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class LegacyCncWorldModelConfig:
    raw_feature_csv: Path
    manifest_csv: Path
    output_dir: Path
    horizons: tuple[int, ...] = (1, 3)
    window_length: int = 8
    stride: int = 1
    failure_horizon_cycles: int = 10
    seed: int = 0
    epochs: int = 6
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 64
    hidden_dim: int = 128
    context_dim: int = 32
    horizon_dim: int = 16
    forecast_loss_weight: float = 0.25
    failure_loss_weight: float = 0.1
    ema_decay: float = 0.98
    claim_scope: str = "experimental_world_model_smoke_not_claim_bearing"
    unsafe_debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_feature_csv", Path(self.raw_feature_csv))
        object.__setattr__(self, "manifest_csv", Path(self.manifest_csv))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.window_length <= 1:
            raise ValueError("window_length must be greater than 1.")
        if not self.horizons:
            raise ValueError("At least one horizon is required.")
        if min(self.horizons) <= 0:
            raise ValueError("Horizons must be positive.")


@dataclass(frozen=True)
class CncWorldModelArrays:
    x: Any
    context: Any
    horizon_index: Any
    target: Any
    failure: Any
    split: Any
    feature_names: tuple[str, ...]
    context_names: tuple[str, ...]
    horizons: tuple[int, ...]
    train_sensor_mean: Any
    train_sensor_std: Any
    train_context_mean: Any
    train_context_std: Any

    @property
    def input_channels(self) -> int:
        return len(self.feature_names)

    @property
    def context_channels(self) -> int:
        return len(self.context_names)


def _seed_everything(seed: int) -> None:
    np, torch, *_ = _require_world_model_dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _split_map(manifest_csv: Path) -> dict[str, str]:
    adapter = LegacyCncWindowsAdapter(manifest_csv)
    return adapter.build_grouped_splits().group_to_split


def _standardize(values: Any, mean_values: Any, std_values: Any) -> Any:
    np = _require_world_model_dependencies()[0]
    return np.nan_to_num((values - mean_values) / std_values, nan=0.0, posinf=0.0, neginf=0.0)


def build_legacy_cnc_world_model_arrays(config: LegacyCncWorldModelConfig) -> CncWorldModelArrays:
    """Build leakage-safe multi-horizon transition arrays from raw CNC feature rows."""

    np = _require_world_model_dependencies()[0]
    adapter = LegacyCncWindowsAdapter(config.manifest_csv)
    adapter.run_leakage_audit().assert_training_allowed(config.unsafe_debug)
    df = load_legacy_cnc_feature_table(config.raw_feature_csv)
    df = df.dropna(subset=["ToolIndex", "CycleToFailure", "NumberOfCycle"]).copy()
    feature_names = tuple(select_feature_columns(df, "sensor_only"))
    context_names = tuple(
        column for column in CONTEXT_COLUMNS if column in df.columns and column not in FORBIDDEN_COLUMNS
    )
    if not context_names:
        raise RuntimeError("No context columns found for CNC world model.")
    split_by_tool = _split_map(config.manifest_csv)
    split_code = {"train": 0, "validation": 1, "test": 2}
    xs: list[Any] = []
    contexts: list[Any] = []
    horizon_indices: list[int] = []
    targets: list[Any] = []
    failures: list[int] = []
    splits: list[int] = []

    horizon_to_index = {horizon: index for index, horizon in enumerate(config.horizons)}
    for tool_id, group in df.groupby("ToolIndex"):
        split = split_by_tool.get(str(int(tool_id)))
        if split not in split_code:
            continue
        group = group.sort_values(["NumberOfCycle", "FileName"]).reset_index(drop=True)
        values = group.loc[:, feature_names].to_numpy(dtype=np.float32)
        context_values = group.loc[:, context_names].to_numpy(dtype=np.float32)
        ctf = group["CycleToFailure"].to_numpy(dtype=float)
        if len(group) < config.window_length + max(config.horizons):
            continue
        starts = range(0, len(group) - config.window_length - max(config.horizons) + 1, config.stride)
        for start in starts:
            end = start + config.window_length
            current = values[start:end]
            context = context_values[end - 1]
            for horizon in config.horizons:
                target_start = start + horizon
                target_end = target_start + config.window_length
                if target_end > len(group):
                    continue
                xs.append(current)
                contexts.append(context)
                horizon_indices.append(horizon_to_index[horizon])
                targets.append(values[target_start:target_end])
                failures.append(int(ctf[target_end - 1] <= config.failure_horizon_cycles))
                splits.append(split_code[split])
    if not xs:
        raise RuntimeError("No CNC world-model transitions were created.")
    x = np.stack(xs).astype(np.float32)
    context = np.stack(contexts).astype(np.float32)
    target = np.stack(targets).astype(np.float32)
    horizon_index = np.asarray(horizon_indices, dtype=np.int64)
    failure = np.asarray(failures, dtype=np.float32)
    split = np.asarray(splits, dtype=np.int64)
    train_mask = split == split_code["train"]
    if not train_mask.any():
        raise RuntimeError("No train transitions after manifest split.")
    train_sensor = np.concatenate([x[train_mask], target[train_mask]], axis=0)
    sensor_mean = np.nanmean(train_sensor, axis=(0, 1), keepdims=True).astype(np.float32)
    sensor_std = np.nanstd(train_sensor, axis=(0, 1), keepdims=True).astype(np.float32)
    sensor_std = np.where(sensor_std < 1e-6, 1.0, sensor_std).astype(np.float32)
    context_mean = np.nanmean(context[train_mask], axis=0, keepdims=True).astype(np.float32)
    context_std = np.nanstd(context[train_mask], axis=0, keepdims=True).astype(np.float32)
    context_std = np.where(context_std < 1e-6, 1.0, context_std).astype(np.float32)
    return CncWorldModelArrays(
        x=_standardize(x, sensor_mean, sensor_std),
        context=_standardize(context, context_mean, context_std),
        horizon_index=horizon_index,
        target=_standardize(target, sensor_mean, sensor_std),
        failure=failure,
        split=split,
        feature_names=feature_names,
        context_names=context_names,
        horizons=config.horizons,
        train_sensor_mean=sensor_mean,
        train_sensor_std=sensor_std,
        train_context_mean=context_mean,
        train_context_std=context_std,
    )


class SensorWindowEncoder(_require_world_model_dependencies()[1].nn.Module):  # type: ignore[misc]
    """Small Conv1D encoder for feature windows."""

    def __init__(self, input_channels: int, embedding_dim: int, hidden_dim: int) -> None:
        torch = _require_world_model_dependencies()[1]
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x.transpose(1, 2))


class DeterministicSensorJepa(_require_world_model_dependencies()[1].nn.Module):  # type: ignore[misc]
    """Context-conditioned deterministic multi-horizon Sensor-JEPA."""

    def __init__(
        self,
        input_channels: int,
        context_channels: int,
        horizon_count: int,
        window_length: int,
        embedding_dim: int,
        hidden_dim: int,
        context_dim: int,
        horizon_dim: int,
    ) -> None:
        torch = _require_world_model_dependencies()[1]
        super().__init__()
        self.encoder = SensorWindowEncoder(input_channels, embedding_dim, hidden_dim)
        self.target_encoder = SensorWindowEncoder(input_channels, embedding_dim, hidden_dim)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.context_encoder = torch.nn.Sequential(
            torch.nn.Linear(context_channels, context_dim),
            torch.nn.GELU(),
            torch.nn.Linear(context_dim, context_dim),
        )
        self.horizon_embedding = torch.nn.Embedding(horizon_count, horizon_dim)
        predictor_in = embedding_dim + context_dim + horizon_dim
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(predictor_in, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )
        self.forecast_head = torch.nn.Linear(embedding_dim, window_length * input_channels)
        self.failure_head = torch.nn.Linear(embedding_dim, 1)
        self.window_length = window_length
        self.input_channels = input_channels

    def predict_latent(self, x: Any, context: Any, horizon_index: Any) -> Any:
        torch = _require_world_model_dependencies()[1]
        z = self.encoder(x)
        c = self.context_encoder(context)
        h = self.horizon_embedding(horizon_index)
        return self.predictor(torch.cat([z, c, h], dim=-1))

    def forward(self, x: Any, context: Any, horizon_index: Any, target: Any | None = None) -> dict[str, Any]:
        torch = _require_world_model_dependencies()[1]
        pred = self.predict_latent(x, context, horizon_index)
        forecast = self.forecast_head(pred).reshape(-1, self.window_length, self.input_channels)
        failure_logit = self.failure_head(pred).squeeze(-1)
        output = {"pred_latent": pred, "forecast": forecast, "failure_logit": failure_logit}
        if target is not None:
            with torch.no_grad():
                target_z = self.target_encoder(target)
            output["target_latent"] = target_z
        return output

    def update_target_encoder(self, decay: float) -> None:
        with _require_world_model_dependencies()[1].no_grad():
            for target_parameter, online_parameter in zip(
                self.target_encoder.parameters(),
                self.encoder.parameters(),
                strict=True,
            ):
                target_parameter.data.mul_(decay).add_(online_parameter.data, alpha=1.0 - decay)


def _batch_indices(indices: Any, batch_size: int, shuffle: bool, seed: int) -> list[Any]:
    np = _require_world_model_dependencies()[0]
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def _device_from_plan(worker_plan: WorkerPlan) -> str:
    torch = _require_world_model_dependencies()[1]
    if worker_plan.device.startswith("cuda") and torch.cuda.is_available():
        return worker_plan.device
    return "cpu"


def _loss_payload(
    model: DeterministicSensorJepa,
    batch: tuple[Any, Any, Any, Any, Any],
    config: LegacyCncWorldModelConfig,
) -> dict[str, Any]:
    torch = _require_world_model_dependencies()[1]
    xb, cb, hb, tb, yb = batch
    out = model(xb, cb, hb, tb)
    latent_loss = torch.nn.functional.mse_loss(out["pred_latent"], out["target_latent"])
    forecast_loss = torch.nn.functional.mse_loss(out["forecast"], tb)
    failure_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], yb)
    loss = (
        latent_loss
        + config.forecast_loss_weight * forecast_loss
        + config.failure_loss_weight * failure_loss
    )
    return {
        "loss": loss,
        "latent_loss": latent_loss.detach(),
        "forecast_loss": forecast_loss.detach(),
        "failure_loss": failure_loss.detach(),
    }


def _evaluate_split(
    model: DeterministicSensorJepa,
    arrays: CncWorldModelArrays,
    split_code: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    np, torch, average_precision_score, f1_score, roc_auc_score = _require_world_model_dependencies()
    indices = np.flatnonzero(arrays.split == split_code)
    model.eval()
    forecast_errors: list[Any] = []
    probabilities: list[Any] = []
    labels: list[Any] = []
    with torch.no_grad():
        for batch_index in _batch_indices(indices, batch_size, shuffle=False, seed=0):
            xb = torch.tensor(arrays.x[batch_index], dtype=torch.float32, device=device)
            cb = torch.tensor(arrays.context[batch_index], dtype=torch.float32, device=device)
            hb = torch.tensor(arrays.horizon_index[batch_index], dtype=torch.long, device=device)
            tb = torch.tensor(arrays.target[batch_index], dtype=torch.float32, device=device)
            out = model(xb, cb, hb)
            per_sample = torch.mean((out["forecast"] - tb) ** 2, dim=(1, 2))
            forecast_errors.append(per_sample.cpu().numpy())
            probabilities.append(torch.sigmoid(out["failure_logit"]).cpu().numpy())
            labels.append(arrays.failure[batch_index])
    y_true = np.concatenate(labels) if labels else np.empty((0,), dtype=np.float32)
    prob = np.concatenate(probabilities) if probabilities else np.empty((0,), dtype=np.float32)
    mse = float(np.concatenate(forecast_errors).mean()) if forecast_errors else math.nan
    if len(set(int(value) for value in y_true)) < 2:
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(y_true, prob))
        auprc = float(average_precision_score(y_true, prob))
    pred = (prob >= 0.5).astype(int)
    return {
        "samples": int(len(indices)),
        "forecast_mse": mse,
        "failure_auroc": auroc,
        "failure_auprc": auprc,
        "failure_f1_at_0_5": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else 0.0,
        "failure_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }


def train_legacy_cnc_world_model(
    config: LegacyCncWorldModelConfig,
    worker_plan: WorkerPlan,
) -> dict[str, Any]:
    """Train and evaluate an experimental deterministic Sensor-JEPA model."""

    np, torch, *_ = _require_world_model_dependencies()
    _seed_everything(config.seed)
    arrays = build_legacy_cnc_world_model_arrays(config)
    device = _device_from_plan(worker_plan)
    model = DeterministicSensorJepa(
        input_channels=arrays.input_channels,
        context_channels=arrays.context_channels,
        horizon_count=len(config.horizons),
        window_length=config.window_length,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        context_dim=config.context_dim,
        horizon_dim=config.horizon_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_indices = np.flatnonzero(arrays.split == 0)
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "latent_loss": 0.0, "forecast_loss": 0.0, "failure_loss": 0.0}
        seen = 0
        for batch_index in _batch_indices(
            train_indices,
            config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            xb = torch.tensor(arrays.x[batch_index], dtype=torch.float32, device=device)
            cb = torch.tensor(arrays.context[batch_index], dtype=torch.float32, device=device)
            hb = torch.tensor(arrays.horizon_index[batch_index], dtype=torch.long, device=device)
            tb = torch.tensor(arrays.target[batch_index], dtype=torch.float32, device=device)
            yb = torch.tensor(arrays.failure[batch_index], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            losses = _loss_payload(model, (xb, cb, hb, tb, yb), config)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target_encoder(config.ema_decay)
            batch_size = int(len(batch_index))
            seen += batch_size
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_size
        history.append({"epoch": epoch, **{key: value / max(seen, 1) for key, value in totals.items()}})

    split_metrics = {
        "train": _evaluate_split(model, arrays, 0, config.batch_size, device),
        "validation": _evaluate_split(model, arrays, 1, config.batch_size, device),
        "test": _evaluate_split(model, arrays, 2, config.batch_size, device),
    }
    warnings = []
    test_metrics = split_metrics["test"]
    if test_metrics["failure_auroc"] == 1.0 or test_metrics["failure_auprc"] == 1.0:
        warnings.append("perfect_test_metric_detected_manual_protocol_review_required")
    report = {
        "experiment_name": "legacy_cnc_deterministic_sensor_jepa_world_model",
        "status": "experimental",
        "hypothesis": (
            "A deterministic context-conditioned multi-horizon sensor world model can predict "
            "future CNC feature windows and provide a failure-soon readout under the audited "
            "held-out-tool split."
        ),
        "dataset": "legacy_mvp_cnc_windows",
        "dataset_version": "industrial_jepa_mvp_54bf4099",
        "split_protocol": "legacy_held_out_tool_manifest",
        "seed": config.seed,
        "input_modalities": ["engineered_sensor_features", "static_context"],
        "action_columns": [],
        "context_columns": list(arrays.context_names),
        "outcome_columns": ["failure_soon_from_cycle_to_failure_for_training_head"],
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "horizons": list(config.horizons),
        "metrics": ["forecast_mse", "failure_auroc", "failure_auprc", "failure_f1_at_0_5"],
        "threshold_selection": "fixed_0_5_for_failure_head_no_test_tuning",
        "test_set_influenced_model_selection": False,
        "claim_scope": config.claim_scope,
        "device": device,
        "worker_plan": worker_plan.to_dict(),
        "history": history,
        "split_metrics": split_metrics,
        "feature_count": arrays.input_channels,
        "context_count": arrays.context_channels,
        "elapsed_seconds": time.time() - start,
        "warnings": warnings,
        "limitations": [
            "This is an experimental deterministic model, not validated production behavior.",
            "Static process settings are context; no action-conditioned or causal claim is made.",
            "Failure labels are used only as supervised train outcomes/readouts, not inputs.",
            "Perfect metrics, if present, require manual leakage/protocol review.",
        ],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "feature_names": arrays.feature_names,
            "context_names": arrays.context_names,
            "horizons": arrays.horizons,
            "train_sensor_mean": arrays.train_sensor_mean,
            "train_sensor_std": arrays.train_sensor_std,
            "train_context_mean": arrays.train_context_mean,
            "train_context_std": arrays.train_context_std,
        },
        checkpoint_path,
    )
    report["checkpoint_path"] = str(checkpoint_path)
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


def load_checkpoint_for_inference(checkpoint_path: Path, device: str = "cpu") -> DeterministicSensorJepa:
    torch = _require_world_model_dependencies()[1]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = DeterministicSensorJepa(
        input_channels=len(checkpoint["feature_names"]),
        context_channels=len(checkpoint["context_names"]),
        horizon_count=len(checkpoint["horizons"]),
        window_length=int(config["window_length"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        context_dim=int(config["context_dim"]),
        horizon_dim=int(config["horizon_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
