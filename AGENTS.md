# PredictiveOps World Model: Early-warning AI for industrial quality, maintenance and operational risk


# AGENTS.md — Engineering and Research Contract

This file defines how human and AI contributors must work in this repository. It is normative. When a task conflicts with this file, stop and document the conflict rather than silently weakening the protocol.

## 1. Mission

Build **ForgeWorld / ACME-JEPA**, a vendor-neutral adaptive industrial world model that:

- learns predictive latent states from industrial multimodal sequences;
- conditions on verified machine and process actions;
- predicts sensor, PLC, quality, failure and throughput outcomes at multiple horizons;
- adapts to individual assets without absorbing progressive degradation into “normal”;
- models propagation from components to machines, stations and lines;
- supports uncertainty-aware, bounded, advisory counterfactual planning;
- is validated more rigorously than conventional anomaly-detection demos.

The project optimizes for industrial usefulness and scientific truth, not for maximizing architectural novelty or marketing claims.

---

## 2. Non-negotiable rules

### 2.1 Never claim SOTA without passing the gate

Do not write “SOTA,” “state of the art,” “best,” “superior,” or equivalent unless `sota_gate.json` passes all relevant checks.

Allowed before the gate:

- “SOTA-target architecture”;
- “candidate method”;
- “competitive under this internal protocol”;
- “improved over baseline X on split Y”;
- “not yet externally validated.”

Every result statement must identify:

- dataset and version;
- split/protocol;
- metric;
- number of seeds;
- baseline;
- uncertainty;
- whether the test set influenced model selection.

### 2.2 Actions are not metadata

A feature is an `action` only when it is a time-aligned command or intervention that could be changed by a controller or operator.

Examples of valid actions:

- feed/RPM setpoint changes;
- overrides;
- actuator commands;
- valve/pump commands;
- robot motion commands;
- recipe setpoints issued at a known timestamp.

Examples of context:

- tool identity;
- material/hardness;
- product family;
- machine identity;
- shift;
- ambient conditions;
- cycle index;
- current recipe identifier.

Never describe a context-conditioned model as causal or action-conditioned.

### 2.3 Never normalize degradation away

Online adaptation must use the dual architecture:

- frozen healthy reference;
- adaptive shadow model;
- versioned adapter state;
- update trust region;
- uncertainty/update gate;
- replay or anti-forgetting mechanism;
- explicit maintenance reset.

A code path that updates the only healthy model online is prohibited.

### 2.4 Prevent leakage by construction

Prohibited as model input when predicting an earlier event:

- future wear;
- cycle-to-failure;
- final quality label;
- future alarm;
- future maintenance action;
- any aggregate computed using future samples;
- random row splits across the same lifecycle/tool/asset.

Every dataset adapter must implement:

```python
build_manifest()
validate_timestamps()
classify_columns()  # action/context/observation/outcome/id
build_grouped_splits()
run_leakage_audit()
```

A training script must refuse to start if the leakage audit fails unless `--unsafe-debug` is explicitly provided. Unsafe-debug outputs must be watermarked and cannot enter benchmark reports.

### 2.5 Simple baselines are mandatory

No neural model result is meaningful without relevant simple baselines.

At minimum:

- persistence or seasonal naive;
- linear/logistic model;
- tree-based model;
- engineered physical features;
- appropriate 1D CNN/TCN/GRU;
- ROCKET-family baseline where applicable;
- PatchCore/PaDiM/EfficientAD for visual anomaly detection.

Do not remove a baseline because it beats the proposed model.

### 2.6 Do not invent data, metrics or implementation status

Never fabricate:

- downloaded datasets;
- sample counts;
- licenses;
- benchmark results;
- passed tests;
- hardware timing;
- production deployments;
- paper conclusions.

Mark planned modules as `planned`, incomplete modules as `experimental`, and completed/reproduced modules as `validated` only after tests and reports exist.

---

## 3. Sources of truth

Priority order:

1. immutable raw data and checksums;
2. machine-readable manifests and split files;
3. experiment configuration;
4. generated metrics/artifacts;
5. code;
6. README and prose reports.

If prose conflicts with generated artifacts, fix the prose. Never edit result CSVs by hand.

Key files:

```text
README.md                         product/research specification
AGENTS.md                         contributor contract
configs/                          experiment truth
 data/manifests/datasets.yaml      dataset truth
 data/manifests/licenses.yaml      licensing truth
 data/splits/                      split truth
 outputs/**/config_resolved.yaml   run truth
 outputs/**/metrics.json           result truth
 outputs/**/sota_gate.json         claim truth
```

---

## 4. Required development workflow

For every substantive change:

1. Read the relevant section of `README.md` and existing configs.
2. Inspect current tests and reports.
3. State the hypothesis being tested.
4. Implement the smallest change that can falsify the hypothesis.
5. Add unit tests and one integration/smoke test.
6. Run leakage and split audits for data changes.
7. Run the relevant simple baselines.
8. Use at least three seeds during development; use the required final seed count for benchmark claims.
9. Generate a machine-readable report.
10. Update documentation with limitations and failed experiments.

Do not start by scaling a model. Establish that the signal and protocol are valid first.

---

## 5. Architecture boundaries

### 5.1 Data plane

Responsible for:

- connectors and file readers;
- timestamp normalization;
- units;
- event synchronization;
- operation/cycle/part linkage;
- missingness and quality flags;
- immutable manifests.

It must not contain model-specific feature engineering beyond documented preprocessing.

### 5.2 Semantic layer

Map raw tags into a canonical schema aligned where possible with:

- OPC UA;
- OPC UA for Machine Tools;
- MTConnect;
- ISA-95 asset hierarchy.

Unknown tags remain explicit `unknown` values; do not guess semantics from names without a mapping record.

### 5.3 Tokenizers

Each tokenizer must expose:

```python
TokenBatch(
    values,
    timestamps,
    semantic_ids,
    asset_ids,
    masks,
    units,
    metadata,
)
```

Tokenizers must support irregular sampling and missing modalities.

### 5.4 Encoders

Encoders produce representations, not decisions. They must expose frozen-feature evaluation and collapse diagnostics.

### 5.5 World models

A module may be called a `world_model` only if it predicts future state conditioned on historical state and, when the claim requires it, real actions.

Required interfaces:

```python
encode(observation_history) -> LatentState
predict(latent_state, actions, context, horizon) -> PredictiveDistribution
score(prediction, observation) -> SurpriseBreakdown
```

### 5.6 Adaptation

Adaptation code must never update the frozen healthy reference. It must log:

- parameters updated;
- learning rate;
- transition IDs;
- pre/post loss;
- adapter norm;
- uncertainty;
- whether update was accepted or rejected.

### 5.7 Decision/advisory layer

The model may rank interventions but cannot write to industrial controllers by default.

Any control integration requires:

- explicit `advisory_only: false` configuration;
- approved action bounds;
- safety interlock design;
- plant owner authorization;
- rollback and audit logs;
- separate security review.

---

## 6. Dataset rules

### 6.1 Downloading

- Prefer owner/official URLs.
- Do not commit raw datasets.
- Do not automatically bypass request forms or terms.
- Record URL, access date, license, checksum and extraction instructions.
- If the official source is unavailable, mark the dataset unavailable; do not silently substitute an unverified mirror.

### 6.2 Splitting

Default priority:

1. held-out site;
2. held-out asset;
3. held-out tool/component;
4. held-out material/recipe;
5. chronological future;
6. grouped cross-validation.

Random row splitting is permitted only for a clearly labeled non-temporal sanity test and must not be used for claims.

### 6.3 Labels and censoring

- Preserve unknown and censored outcomes.
- Do not label preventive replacement as exact failure time.
- Use survival methods when failure has not been observed.
- Maintenance events create state boundaries; they are not ordinary samples.

### 6.4 Public-versus-pilot data

Public datasets establish reproducibility and module competence. Production claims require real pilot data. Keep these conclusions separate in reports.

---

## 7. Experiment protocols

Every experiment configuration must include:

```yaml
experiment_name:
hypothesis:
dataset:
dataset_version:
split_protocol:
seed:
input_modalities:
action_columns:
context_columns:
outcome_columns:
forbidden_columns:
model:
baselines:
metrics:
threshold_selection:
compute:
claim_scope:
```

### 7.1 Required ablations for action-conditioned models

- no actions;
- context only;
- correct actions;
- shuffled actions;
- action-only;
- current state only;
- future predictor;
- action dropout;
- horizon grid.

If shuffled actions perform as well as correct actions, do not claim causal action modeling.

### 7.2 Required ablations for adaptation

- frozen model;
- full fine-tuning;
- last-layer/adapter only;
- standard test-time adaptation baseline;
- unrestricted adaptation;
- dual healthy/shadow adaptation;
- replay/no replay;
- trust region/no trust region;
- benign shift versus progressive degradation.

### 7.3 Required ablations for multimodal fusion

- each modality alone;
- all pairs where feasible;
- full fusion;
- missing-modality test;
- corrupted-modality test;
- time-misalignment sensitivity.

### 7.4 Required ablations for generative branches

- deterministic JEPA predictor;
- probabilistic latent predictor;
- generative RAE/diffusion branch;
- equal compute where possible;
- planning/diagnostic value, not only visual quality.

---

## 8. Metrics and threshold policy

### 8.1 Never optimize only AUROC

Industrial reporting must include:

- AUPRC for imbalanced labels;
- event-based precision/recall;
- false alarms per hour or 1,000 cycles;
- lead time;
- missed events;
- calibration;
- per-asset and worst-group performance.

### 8.2 Threshold selection

Thresholds and anomaly-score direction are selected on validation data only. Test labels cannot be used to:

- invert a score;
- choose a percentile;
- tune smoothing;
- choose a persistence window;
- select the best horizon.

### 8.3 Quality and RUL

Quality forecasts require operational metrics such as first-out-of-tolerance lead time. RUL requires censoring-aware metrics and calibration, not only RMSE.

---

## 9. SOTA gate implementation

`src/forgeworld/evaluation/sota_gate.py` must evaluate:

```yaml
protocol:
  official_split: true|false
  leakage_pass: true|false
  validation_only_selection: true|false
baselines:
  exact_strong_baselines: true|false
  simple_baselines: true|false
statistics:
  required_seeds_met: true|false
  confidence_intervals: true|false
  significance_test: true|false
generalization:
  held_out_asset_or_site: true|false
  cross_dataset_or_pilot: true|false
operations:
  lead_time_reported: true|false
  false_alarm_rate_reported: true|false
  latency_reported: true|false
  calibration_reported: true|false
reproducibility:
  config_saved: true|false
  checksums_saved: true|false
  code_commit_saved: true|false
```

A missing field is a failure, not “not applicable,” unless a written metric-specific exemption exists.

---

## 10. Coding standards

- Python 3.11+.
- Type hints on public interfaces.
- `ruff`, `mypy` and `pytest` in CI.
- No hidden global state.
- No hard-coded local paths.
- Config-driven experiments.
- Deterministic seeds recorded.
- Use `pathlib`.
- Use structured logging, not `print`, in library code.
- Use dataclasses or typed models for data contracts.
- Keep model, data and evaluation logic separate.
- Avoid unnecessary dependencies.
- Provide CPU smoke mode for every training/evaluation command.
- GPU-only behavior must fail with an actionable error.

### 10.1 Numerical and physical handling

- Store raw values and units.
- Fit normalization on train only.
- Persist normalization parameters.
- Handle NaNs explicitly.
- Do not interpolate across maintenance boundaries or machine-off periods.
- Do not mix operating modes without mode/context tokens.

### 10.2 Security

- Never embed credentials.
- Redact plant identifiers from public reports.
- Validate serialized model and dataset inputs.
- Treat OPC UA/MQTT/Kafka connectors as untrusted input surfaces.
- Online model updates require signed/versioned artifacts in production designs.

---

## 11. Testing requirements

### Unit tests

- token shapes and masks;
- timestamp alignment;
- unit conversion;
- action/context classification;
- leakage detection;
- maintenance-boundary handling;
- metric correctness;
- adapter freeze guarantees.

### Integration tests

- one end-to-end public dataset quick run;
- checkpoint save/load;
- missing modality;
- CPU inference;
- deterministic repeat;
- report generation.

### Adversarial tests

- shuffled timestamps;
- duplicated future labels;
- hidden cycle-to-failure column;
- mislabeled action column;
- changing units;
- sensor dropout;
- recipe shift;
- progressive degradation during adaptation.

The test suite must explicitly verify that the healthy model parameters never change.

---

## 12. Documentation requirements

Every implemented dataset must update:

- `README.md` dataset status;
- manifest and license file;
- data card;
- download/preparation command;
- split rationale;
- known leakage risks.

Every model must include:

- intended use;
- architecture summary;
- training objective;
- required modalities;
- limitations;
- compute profile;
- validated datasets;
- unsupported claims.

Every benchmark report must include a “What this result does not prove” section.

---

## 13. Research priorities

Prioritize in this order:

1. protocol correctness and leakage prevention;
2. strong baselines;
3. real action data;
4. predictive value and lead time;
5. dual adaptation safety;
6. quality linkage;
7. multimodal fusion;
8. line graph;
9. generative visualization;
10. model scale.

Do not spend major compute on MIRA-scale generation before the deterministic latent world model proves incremental industrial value.

---

## 14. Definition of done by phase

### Dataset adapter done

- official source/terms recorded;
- checksum recorded;
- train/validation/test split generated;
- data card written;
- leakage audit passes;
- one loader test passes.

### Model experiment done

- hypothesis specified;
- baselines run;
- config and seed saved;
- metrics and per-group results saved;
- runtime measured;
- failure cases inspected;
- report generated;
- claim scope documented.

### Adaptation module done

- healthy reference immutable by test;
- accepted/rejected updates logged;
- benign shift improves;
- progressive degradation remains detectable;
- rollback works;
- compute budget measured.

### Pilot-ready done

- read-only connector;
- data-quality monitoring;
- shadow inference;
- alert provenance;
- false-alarm and lead-time dashboard;
- rollback;
- security review;
- operator review protocol.

---

## 15. Expected contributor behavior

Contributors should actively look for evidence that the proposed architecture is unnecessary or wrong.

Good contributions include:

- finding that a simple baseline wins;
- identifying leakage;
- showing that an “action” is only context;
- documenting a negative result;
- reducing model size without losing value;
- improving calibration or lead time without improving AUROC;
- proving that a modality adds no incremental information;
- narrowing a claim.

Bad contributions include:

- optimizing the test set;
- hiding weak assets/categories in an average;
- calling an internal split SOTA;
- replacing an exact baseline with a weaker homemade approximation without disclosure;
- using future lifecycle variables;
- adapting the healthy model online;
- implementing uncontrolled PLC writes;
- fabricating progress in documentation.

---

## 16. First tasks for a coding agent

Unless a more specific issue is assigned, work in this order:

1. Preserve the current MVP as a reproducible `legacy_mvp` benchmark.
2. Create the canonical data contracts and action/context/outcome validator.
3. Implement official Priority A dataset manifests and download instructions.
4. Build grouped split and leakage tests.
5. Reproduce strong sensor and visual baselines.
6. Implement a deterministic multi-horizon Sensor-JEPA world model.
7. Add SWaT/TEP PLC event prediction with correct-action versus shuffled-action tests.
8. Implement frozen-healthy plus adaptive-shadow adapters.
9. Link one sensor dataset to a quality/degradation target.
10. Only then begin multimodal and graph modeling.

At the end of every task, state:

- what changed;
- tests run;
- evidence produced;
- limitations;
- next falsifiable hypothesis.
