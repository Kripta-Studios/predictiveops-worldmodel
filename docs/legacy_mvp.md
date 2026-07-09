# Legacy MVP Benchmark Reference

This repository starts ForgeWorld from the existing Industrial JEPA MVP without
copying raw datasets or asserting new benchmark results.

## Provenance

- Source repository: `../industrial_jepa_mvp` or `FORGEWORLD_LEGACY_MVP_ROOT`.
- Source commit inspected: `54bf4099f36dfbc69e6cc4da76ab6c62499c056d`.
- Source worktree status at inspection: clean.
- Source manifest used: `industrial_jepa_mvp/data/manifests/datasets.yaml`.

## Preserved Commands

```powershell
python scripts/00_create_dataset_manifest.py
python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml
python scripts/20_train_sensor_world_model.py --config configs/sensor_jepa/demo_sensor_quick.yaml
python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml
python scripts/17_run_all_demos.py
```

## Current Limitations

- This is a provenance descriptor, not a reproduced ForgeWorld benchmark report.
- Dataset licenses and checksums still need owner-level verification in this repo.
- Legacy split and threshold logic must pass ForgeWorld leakage audits before use
  in benchmark reports.
- No state-of-the-art claim is permitted from this descriptor.

## CNC Window Audit

The first ForgeWorld adapter audits the legacy `cnc_windows.csv` manifest:

```powershell
python scripts/data/audit_legacy_cnc_windows.py
```

The adapter preserves the held-out-tool split in the manifest and classifies
`CycleToFailure` plus `CycleToFailureNormalized` as forbidden inputs. It also
classifies the legacy process settings such as `FeedRate`, `ToolRotation`,
`ADOC`, and `RDOC` as context, not actions, because the manifest does not prove
time-aligned controller commands.

## CNC Baseline Protocols

The simple baseline runner now supports two grouped split protocols:

- `legacy_held_out_tool_manifest`: preserves the legacy held-out-tool manifest.
- `held_out_cutting_condition`: holds out complete `ADOC`/`RDOC` cutting-condition
  groups to test whether engineered sensor features generalize beyond seen static
  process context.

Example smoke run:

```powershell
python scripts/benchmarks/run_legacy_cnc_baselines.py --smoke --split-protocol held_out_cutting_condition --output-dir outputs/baselines/legacy_cnc_failure_soon_held_out_cutting_condition
```

The generated `metrics.json` summarizes metrics and groups; the full row-level
split assignments are written separately to `split_manifest.json`. This result
does not prove causal action modeling, because `ADOC` and `RDOC` remain context
columns in this legacy data.

## MVTec Bottle Visual Baseline

The legacy MVTec AD bottle manifest is now audited with the same non-leakage
contract. `label` and `defect_type` are outcomes, and pixel masks are forbidden
inputs for image-level baselines.

Audit command:

```powershell
python scripts/data/audit_legacy_mvtec_bottle.py
```

Example smoke run:

```powershell
python scripts/benchmarks/run_legacy_mvtec_baselines.py --smoke --output-dir outputs/baselines/legacy_mvtec_bottle_simple
```

The baseline uses nearest-neighbor distance over simple image features and
selects the alert threshold from held-out official train-good images only. This
is a visual sanity floor; it does not replace PatchCore, PaDiM, EfficientAD, or
frozen foundation-feature baselines required for serious visual comparisons.

## Next Falsifiable Hypothesis

If the legacy CNC and MVTec tasks are rerun through the new schema, split, and
leakage gates, then any unsafe feature, threshold, or group split should fail
before training starts.
