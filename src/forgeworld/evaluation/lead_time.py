"""Event lead-time and false-alarm metrics for industrial sequences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EventLeadTimeMetrics:
    events_total: int
    events_detected: int
    events_missed: int
    event_recall: float | None
    median_lead_cycles: float | None
    mean_lead_cycles: float | None
    false_alarms: int
    false_alarms_per_1000_cycles: float | None
    evaluated_cycles: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "events_total": self.events_total,
            "events_detected": self.events_detected,
            "events_missed": self.events_missed,
            "event_recall": self.event_recall,
            "median_lead_cycles": self.median_lead_cycles,
            "mean_lead_cycles": self.mean_lead_cycles,
            "false_alarms": self.false_alarms,
            "false_alarms_per_1000_cycles": self.false_alarms_per_1000_cycles,
            "evaluated_cycles": self.evaluated_cycles,
        }


def event_lead_time_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    asset_key: str = "asset_id",
    cycle_key: str = "cycle",
    label_key: str = "label",
    alert_key: str = "alert",
) -> EventLeadTimeMetrics:
    """Evaluate first alert before first positive event for each asset.

    A false alarm is an alert before the first event when that asset is missed,
    or any alert on an asset with no event in the evaluated range. Alerts after
    the first event are not counted because they are no longer early warnings.
    """

    by_asset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    total_cycles = 0
    for record in records:
        by_asset[str(record[asset_key])].append(record)
        total_cycles += 1

    lead_times: list[float] = []
    false_alarms = 0
    events_total = 0
    events_detected = 0
    for asset_records in by_asset.values():
        ordered = sorted(asset_records, key=lambda row: float(row[cycle_key]))
        positive = [row for row in ordered if int(row[label_key]) == 1]
        if not positive:
            false_alarms += sum(1 for row in ordered if int(row[alert_key]) == 1)
            continue
        events_total += 1
        first_event_cycle = float(positive[0][cycle_key])
        pre_event_alerts = [
            row
            for row in ordered
            if float(row[cycle_key]) <= first_event_cycle and int(row[alert_key]) == 1
        ]
        if pre_event_alerts:
            first_alert_cycle = float(pre_event_alerts[0][cycle_key])
            lead_times.append(first_event_cycle - first_alert_cycle)
            events_detected += 1
        else:
            false_alarms += sum(
                1
                for row in ordered
                if float(row[cycle_key]) < first_event_cycle and int(row[alert_key]) == 1
            )
    events_missed = events_total - events_detected
    event_recall = events_detected / events_total if events_total else None
    false_alarm_rate = false_alarms / total_cycles * 1000.0 if total_cycles else None
    return EventLeadTimeMetrics(
        events_total=events_total,
        events_detected=events_detected,
        events_missed=events_missed,
        event_recall=event_recall,
        median_lead_cycles=float(median(lead_times)) if lead_times else None,
        mean_lead_cycles=float(sum(lead_times) / len(lead_times)) if lead_times else None,
        false_alarms=false_alarms,
        false_alarms_per_1000_cycles=false_alarm_rate,
        evaluated_cycles=total_cycles,
    )
