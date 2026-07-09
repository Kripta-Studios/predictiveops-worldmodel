"""Priority dataset registry and download policy metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ACCESS_DATE = "2026-07-09"


@dataclass(frozen=True)
class PriorityDataset:
    dataset_id: str
    name: str
    priority: str
    family: str
    status: str
    official_source_url: str | None
    terms_url: str | None
    download_policy: str
    license_status: str
    task_role: str
    modalities: tuple[str, ...]
    required_split_protocols: tuple[str, ...]
    action_context_policy: str
    leakage_risks: tuple[str, ...]
    local_path_hint: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["access_date"] = ACCESS_DATE
        return payload


PRIORITY_A_DATASETS: tuple[PriorityDataset, ...] = (
    PriorityDataset(
        dataset_id="nasa_milling_wear",
        name="NASA Milling Wear",
        priority="A",
        family="sensor",
        status="planned_manifest_only",
        official_source_url="https://data.nasa.gov/dataset/milling-wear",
        terms_url="https://data.nasa.gov/dataset/milling-wear",
        download_policy="owner_url_allowed_record_checksum_after_download",
        license_status="official_page_reports_other_license_specified_needs_review",
        task_role="CNC wear representation and transfer",
        modalities=("force", "vibration_or_acoustic", "process_metadata", "wear_measurement"),
        required_split_protocols=("held_out_tool_or_cutting_condition", "chronological_future"),
        action_context_policy=(
            "feeds, speeds, and depth-of-cut are context unless command timestamps are present"
        ),
        leakage_risks=("wear labels", "cycle-to-failure", "aggregates over full tool life"),
        local_path_hint="data/raw/sensor/nasa_milling_wear",
        notes=("Official NASA Open Data source should be preferred over mirrors.",),
    ),
    PriorityDataset(
        dataset_id="phm2010_milling",
        name="PHM 2010 Milling Challenge",
        priority="A",
        family="sensor",
        status="planned_manifest_only",
        official_source_url=(
            "https://phmsociety.org/phm_competition/"
            "2010-phm-society-conference-data-challenge/"
        ),
        terms_url=(
            "https://phmsociety.org/phm_competition/"
            "2010-phm-society-conference-data-challenge/"
        ),
        download_policy="owner_url_allowed_record_terms_and_checksum_after_download",
        license_status="terms_need_manual_review",
        task_role="tool wear and RUL under leave-one-cutter-out protocol",
        modalities=("dynamometer", "accelerometer", "acoustic_emission", "wear_per_cut"),
        required_split_protocols=("leave_one_cutter_out", "validation_only_thresholds"),
        action_context_policy="cutting parameters are context unless command events are timestamped",
        leakage_risks=("wear per cut at future horizons", "RUL target leakage", "cutter overlap"),
        local_path_hint="data/raw/sensor/phm2010_milling",
    ),
    PriorityDataset(
        dataset_id="cnc_tool_wear_18_runs",
        name="CNC Mill Tool Wear / 18 runs",
        priority="A",
        family="sensor",
        status="blocked_pending_verified_owner_source",
        official_source_url=None,
        terms_url=None,
        download_policy="do_not_download_from_mirror_until_owner_terms_are_recorded",
        license_status="unverified",
        task_role="action/context audit and cycle segmentation",
        modalities=("controller_trace", "process_time_series", "tool_condition"),
        required_split_protocols=("held_out_run_or_tool", "chronological_future"),
        action_context_policy="controller variables require timestamp audit before action claims",
        leakage_risks=("tool condition labels", "run-level random split", "post-cycle aggregates"),
        local_path_hint="data/raw/sensor/cnc_tool_wear_18_runs",
        notes=("README notes common mirrors; mirrors are not accepted as source of truth.",),
    ),
    PriorityDataset(
        dataset_id="metropt3",
        name="MetroPT-3",
        priority="A",
        family="sensor",
        status="planned_manifest_only",
        official_source_url="https://zenodo.org/records/6854240",
        terms_url="https://zenodo.org/records/6854240",
        download_policy="owner_url_allowed_record_doi_version_and_checksum_after_download",
        license_status="zenodo_record_terms_need_review",
        task_role="real temporal anomaly, lead time, and adaptation",
        modalities=("pressure", "temperature", "current", "valve_state", "maintenance_report"),
        required_split_protocols=("chronological_future", "maintenance_boundary_aware"),
        action_context_policy="valve states are observations/actions only after command provenance audit",
        leakage_risks=("maintenance reports after event", "future fault intervals", "test-label thresholds"),
        local_path_hint="data/raw/sensor/metropt3",
        notes=("Scientific Data paper links the Zenodo record.",),
    ),
    PriorityDataset(
        dataset_id="hydraulic_systems",
        name="Hydraulic System Condition Monitoring",
        priority="A",
        family="sensor",
        status="planned_manifest_only",
        official_source_url=(
            "https://archive.ics.uci.edu/dataset/447/"
            "condition+monitoring+of+hydraulic+systems"
        ),
        terms_url=(
            "https://archive.ics.uci.edu/dataset/447/"
            "condition+monitoring+of+hydraulic+systems"
        ),
        download_policy="owner_url_allowed_record_uci_metadata_and_checksum_after_download",
        license_status="uci_terms_need_review",
        task_role="multi-component degradation and severity",
        modalities=("pressure", "flow", "temperature", "vibration", "component_condition"),
        required_split_protocols=("held_out_component_condition", "grouped_cross_validation"),
        action_context_policy="load cycles and operating profiles are context unless intervention events exist",
        leakage_risks=("condition labels as input", "profile-level random split", "full-cycle aggregates"),
        local_path_hint="data/raw/sensor/hydraulic_systems",
    ),
    PriorityDataset(
        dataset_id="swat",
        name="Secure Water Treatment",
        priority="A",
        family="plc",
        status="manual_request_required",
        official_source_url=(
            "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/"
            "dataset-characteristics/swat/"
        ),
        terms_url=(
            "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/"
            "dataset-characteristics/swat/"
        ),
        download_policy="manual_request_only_do_not_bypass_terms",
        license_status="request_terms_required",
        task_role="hybrid state/action transition modeling",
        modalities=("plc_tags", "scada_sensors", "actuators", "network_pcap"),
        required_split_protocols=("chronological_normal_attack_split", "validation_only_thresholds"),
        action_context_policy="actuator tags may be actions only after command/response semantics are mapped",
        leakage_risks=("attack labels", "future alarms", "timestamp shuffling", "test-label score direction"),
        local_path_hint="data/raw/plc/swat",
    ),
    PriorityDataset(
        dataset_id="mvtec_ad2",
        name="MVTec AD 2",
        priority="A",
        family="visual",
        status="planned_manifest_only",
        official_source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2",
        terms_url="https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2",
        download_policy="owner_url_allowed_respect_evaluation_server_and_terms",
        license_status="mvtec_terms_need_review",
        task_role="current visual anomaly benchmark with held-out evaluation",
        modalities=("rgb_image", "pixel_mask_public_where_available", "evaluation_server"),
        required_split_protocols=("official_mvtec_ad2_split", "public_vs_server_test_separation"),
        action_context_policy="not_action_dataset",
        leakage_risks=("test masks", "category-specific thresholding on test", "public/server mixup"),
        local_path_hint="data/raw/visual/mvtec_ad2",
    ),
    PriorityDataset(
        dataset_id="mvtec_loco_ad",
        name="MVTec LOCO AD",
        priority="A",
        family="visual",
        status="planned_manifest_only",
        official_source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad",
        terms_url="https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad",
        download_policy="owner_url_allowed_record_terms_and_checksum_after_download",
        license_status="mvtec_terms_need_review",
        task_role="logical and structural anomaly detection",
        modalities=("rgb_image", "logical_anomaly_label", "pixel_or_region_annotation"),
        required_split_protocols=("official_train_test_split",),
        action_context_policy="not_action_dataset",
        leakage_risks=("logical test labels", "category thresholding on test", "mask leakage"),
        local_path_hint="data/raw/visual/mvtec_loco_ad",
    ),
    PriorityDataset(
        dataset_id="mvtec_3d_ad",
        name="MVTec 3D-AD",
        priority="A",
        family="visual_3d",
        status="planned_manifest_only",
        official_source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad",
        terms_url="https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad",
        download_policy="owner_url_allowed_record_terms_and_checksum_after_download",
        license_status="mvtec_terms_need_review",
        task_role="geometric defects and multimodal vision",
        modalities=("rgb_image", "depth_or_point_cloud", "pixel_mask"),
        required_split_protocols=("official_train_test_split", "modality_missingness_test"),
        action_context_policy="not_action_dataset",
        leakage_risks=("test masks", "aligned modality leakage", "category thresholding on test"),
        local_path_hint="data/raw/visual/mvtec_3d_ad",
    ),
    PriorityDataset(
        dataset_id="visa",
        name="Visual Anomaly (VisA)",
        priority="A",
        family="visual",
        status="external_local_reference_partial",
        official_source_url="https://registry.opendata.aws/visa/",
        terms_url="https://creativecommons.org/licenses/by/4.0/",
        download_policy="aws_open_data_allowed_record_snapshot_and_checksum_after_download",
        license_status="cc_by_4_0_reported_by_aws_registry",
        task_role="cross-dataset visual generalization",
        modalities=("rgb_image", "image_label", "pixel_annotation"),
        required_split_protocols=("official_or_documented_category_split", "cross_dataset_evaluation"),
        action_context_policy="not_action_dataset",
        leakage_risks=("test masks", "category thresholding on test", "duplicate image paths"),
        local_path_hint="data/raw/visual/visa",
        notes=("Legacy MVP manifest reports partial local VisA files.",),
    ),
)


def get_priority_dataset(dataset_id: str) -> PriorityDataset:
    for dataset in PRIORITY_A_DATASETS:
        if dataset.dataset_id == dataset_id:
            return dataset
    known = ", ".join(dataset.dataset_id for dataset in PRIORITY_A_DATASETS)
    raise KeyError(f"Unknown priority dataset {dataset_id!r}. Known: {known}")
