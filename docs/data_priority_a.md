# Priority A Dataset Plan

Priority A datasets are planned support targets. This document records owner
sources and protocol risks; it does not claim that every dataset is downloaded,
licensed for the intended use, or validated.

## Source Policy

- Prefer owner or official registry URLs.
- Do not commit raw datasets.
- Do not automate request forms or bypass terms.
- If the owner source is unavailable or unclear, mark the dataset blocked rather
  than using a mirror silently.
- Record checksum, archive version, access date, extraction steps, and license
  review before training.

## Download Instructions

```powershell
python scripts/download/requested_dataset_instructions.py swat
python scripts/download/requested_dataset_instructions.py priority_a --json
```

## Current Priority A Status

- `nasa_milling_wear`: planned; official NASA Open Data source recorded.
- `phm2010_milling`: planned; PHM Society source recorded.
- `cnc_tool_wear_18_runs`: blocked until an owner source and terms are verified.
- `metropt3`: planned; Zenodo DOI record and paper source recorded.
- `hydraulic_systems`: planned; UCI source recorded.
- `swat`: manual request required through iTrust.
- `mvtec_ad2`: planned; MVTec owner source and evaluation-server caveat recorded.
- `mvtec_loco_ad`: planned; MVTec owner source recorded.
- `mvtec_3d_ad`: planned; MVTec owner source recorded.
- `visa`: partial local reference exists in the legacy MVP; AWS Open Data source
  and CC BY 4.0 terms URL are recorded.

## Action/Context Rule

For the CNC and PLC datasets, process settings are not treated as actions unless
the adapter can prove they are time-aligned commands or interventions. Static
recipe, tool, material, hardness, and operating-condition fields remain context.
