from __future__ import annotations

from forgeworld.evaluation.lead_time import event_lead_time_metrics


def test_event_lead_time_detects_first_event_and_lead_cycles() -> None:
    records = [
        {"asset_id": "a", "cycle": 1, "label": 0, "alert": 0},
        {"asset_id": "a", "cycle": 2, "label": 0, "alert": 1},
        {"asset_id": "a", "cycle": 5, "label": 1, "alert": 1},
        {"asset_id": "b", "cycle": 1, "label": 0, "alert": 0},
        {"asset_id": "b", "cycle": 2, "label": 1, "alert": 0},
        {"asset_id": "c", "cycle": 1, "label": 0, "alert": 1},
        {"asset_id": "c", "cycle": 2, "label": 0, "alert": 0},
    ]

    metrics = event_lead_time_metrics(records).to_dict()

    assert metrics["events_total"] == 2
    assert metrics["events_detected"] == 1
    assert metrics["events_missed"] == 1
    assert metrics["event_recall"] == 0.5
    assert metrics["median_lead_cycles"] == 3.0
    assert metrics["false_alarms"] == 1
    assert metrics["false_alarms_per_1000_cycles"] == 1000.0 / 7.0


def test_event_lead_time_handles_no_events() -> None:
    records = [
        {"asset_id": "a", "cycle": 1, "label": 0, "alert": 1},
        {"asset_id": "a", "cycle": 2, "label": 0, "alert": 0},
    ]

    metrics = event_lead_time_metrics(records).to_dict()

    assert metrics["events_total"] == 0
    assert metrics["event_recall"] is None
    assert metrics["false_alarms"] == 1
