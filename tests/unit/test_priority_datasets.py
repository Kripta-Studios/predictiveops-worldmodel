from __future__ import annotations

from forgeworld.data.priority import PRIORITY_A_DATASETS, get_priority_dataset


def test_priority_a_registry_has_required_entries_and_sources() -> None:
    ids = {dataset.dataset_id for dataset in PRIORITY_A_DATASETS}

    assert {
        "nasa_milling_wear",
        "phm2010_milling",
        "cnc_tool_wear_18_runs",
        "metropt3",
        "hydraulic_systems",
        "swat",
        "mvtec_ad2",
        "mvtec_loco_ad",
        "mvtec_3d_ad",
        "visa",
    } <= ids
    for dataset in PRIORITY_A_DATASETS:
        if dataset.dataset_id == "cnc_tool_wear_18_runs":
            assert dataset.official_source_url is None
            assert dataset.status == "blocked_pending_verified_owner_source"
        else:
            assert dataset.official_source_url is not None
        assert dataset.required_split_protocols
        assert dataset.leakage_risks


def test_swat_download_policy_requires_manual_request() -> None:
    swat = get_priority_dataset("swat")

    assert swat.download_policy == "manual_request_only_do_not_bypass_terms"
    assert swat.status == "manual_request_required"


def test_visual_datasets_are_not_action_datasets() -> None:
    visual = [
        dataset for dataset in PRIORITY_A_DATASETS if dataset.family in {"visual", "visual_3d"}
    ]

    assert visual
    assert all(dataset.action_context_policy == "not_action_dataset" for dataset in visual)
