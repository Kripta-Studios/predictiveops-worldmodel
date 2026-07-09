from __future__ import annotations

from pathlib import Path

from forgeworld.world_models.sensor_jepa import (
    DeterministicSensorJepa,
    LegacyCncWorldModelConfig,
    load_checkpoint_for_inference,
)


def test_deterministic_sensor_jepa_forward_shapes() -> None:
    import torch

    model = DeterministicSensorJepa(
        input_channels=4,
        context_channels=3,
        horizon_count=2,
        window_length=5,
        embedding_dim=8,
        hidden_dim=16,
        context_dim=6,
        horizon_dim=4,
    )
    x = torch.randn(7, 5, 4)
    context = torch.randn(7, 3)
    horizon = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
    target = torch.randn(7, 5, 4)

    out = model(x, context, horizon, target)

    assert out["pred_latent"].shape == (7, 8)
    assert out["target_latent"].shape == (7, 8)
    assert out["forecast"].shape == (7, 5, 4)
    assert out["failure_logit"].shape == (7,)


def test_world_model_config_validates_horizons(tmp_path: Path) -> None:
    try:
        LegacyCncWorldModelConfig(
            raw_feature_csv=tmp_path / "raw.csv",
            manifest_csv=tmp_path / "manifest.csv",
            output_dir=tmp_path / "out",
            horizons=(0,),
        )
    except ValueError as exc:
        assert "Horizons" in str(exc)
    else:
        raise AssertionError("Expected invalid horizon to raise ValueError")


def test_world_model_checkpoint_round_trip(tmp_path: Path) -> None:
    import torch

    model = DeterministicSensorJepa(
        input_channels=2,
        context_channels=1,
        horizon_count=1,
        window_length=3,
        embedding_dim=4,
        hidden_dim=8,
        context_dim=3,
        horizon_dim=2,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "window_length": 3,
                "embedding_dim": 4,
                "hidden_dim": 8,
                "context_dim": 3,
                "horizon_dim": 2,
            },
            "feature_names": ("a", "b"),
            "context_names": ("c",),
            "horizons": (1,),
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_for_inference(checkpoint_path)

    assert isinstance(loaded, DeterministicSensorJepa)
