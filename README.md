# PredictiveOps World Model: Early-warning AI for industrial quality, maintenance and operational risk


## Adaptive Causal Industrial World Models for Predictive Quality, Maintenance and Production Control

> **Project status:** research and engineering specification with an initial protocol scaffold. The prior Industrial JEPA MVP is preserved as an external `legacy_mvp` benchmark reference; ForgeWorld is the next-generation project described here.
>
> **Claim policy:** ForgeWorld is a **SOTA-target project**, not a SOTA result. No state-of-the-art claim is permitted until the validation gates in [SOTA claim protocol](#sota-claim-protocol) are satisfied.
>
> **Sources last reviewed:** 2026-07-07.

Initial executable scaffolding now includes:

- canonical data-column roles and action/context/outcome validation;
- grouped split and leakage-audit utilities;
- a strict `sota_gate.json` evaluator;
- token and world-model interface contracts;
- hardware-aware compute planning for CPU/GPU runs with CPU smoke mode.
- Priority A dataset manifest and owner-respecting download/request instructions.
- a leakage-guarded CNC simple-baseline runner for majority, logistic, and tree baselines.

---

## 1. Executive summary

Industrial AI products commonly solve one isolated problem:

- detect that a sensor is abnormal;
- compare a CNC against a mechanical fingerprint;
- classify an image as defective;
- forecast a scalar signal;
- estimate remaining useful life for one asset class;
- simulate a known process with a manually built digital twin;
- summarize alarms and maintenance documents with a copilot.

ForgeWorld targets a different capability:

> **Learn a vendor-neutral, multimodal and continuously calibrated model of how an industrial asset or production line evolves under commands, operating conditions and hidden degradation; use it to predict quality drift and failures, explain their propagation, and evaluate safe interventions before they are applied.**

The system is inspired by Joint-Embedding Predictive Architectures (JEPA), DINOv3, V-JEPA 2/2.1, DINO-WM, AdaJEPA and MIRA, but is redesigned for industrial data rather than games or robotics.

Its core prediction is:

\[
p\!\left(
 z_{t+1:t+H},
 y_{t+1:t+H},
 q_{t+1:t+H}
 \mid
 z_{\le t},
 u_{t:t+H},
 c_t,
 G_t
\right),
\]

where:

- \(z_t\) is the hidden health and process state;
- \(u_t\) contains **verified control actions** such as setpoints, overrides and actuator commands;
- \(c_t\) contains non-interventional context such as material, product, tool, shift and ambient conditions;
- \(G_t\) is the asset/line graph;
- \(y_t\) contains future sensor and event observations;
- \(q_t\) contains quality, throughput and failure outcomes.

The first commercial use case is deliberately narrow:

> **Predict tool/asset degradation and the first quality deviation early enough to act, while separating normal recipe changes from true degradation.**

The long-term platform expands from a CNC cell to PLC-controlled stations and then to a production-line graph.

---

## 2. The real industrial problem

A factory rarely fails because one signal crosses one fixed threshold. More often:

1. a tool, bearing, actuator or fixture begins to degrade;
2. the controller compensates, hiding the fault;
3. energy, vibration, timing or tracking error changes slightly;
4. the effect appears only during a specific operation or recipe;
5. the deviation propagates to positioning, force, torque or surface quality;
6. defects appear several cycles later or at another station;
7. operators see multiple alarms but no causal explanation.

Traditional anomaly detection estimates something like:

\[
A_t=d(x_t,\mathcal{D}_{\text{normal}}).
\]

ForgeWorld instead asks:

\[
A_t=d\!\left(
x_{t+1},
\mathbb{E}[x_{t+1}\mid x_{\le t},u_t,c_t,G_t]
\right).
\]

This distinction is essential. A high spindle current can be normal under a harder material or deeper cut and abnormal under an unchanged recipe. A slower valve response may still remain inside a PLC timeout while revealing progressive pneumatic or mechanical degradation. A dimensional defect may be caused upstream by fixture drift rather than by the inspection station that discovers it.

The target output is not merely an anomaly score. A production alert should answer:

- What changed?
- When did the deviation begin?
- Under which operation or command does it appear?
- Is it a recipe/domain shift or degradation?
- Which component or upstream station is the likely origin?
- How uncertain is the conclusion?
- How many cycles remain before quality or availability is affected?
- Which safe intervention is expected to reduce the risk?

---

## 3. Why this is not just another predictive-maintenance product

Siemens currently offers, among other capabilities:

- [Senseye Predictive Maintenance](https://www.siemens.com/en-us/products/industrial-digitalization-services/senseye-predictive-maintenance/), which learns from condition and operational data and supports maintenance workflows;
- [SIMATIC Anomaly Detection](https://www.siemens.com/en-gb/products/simatic-apps/anomaly-detection/), an Industrial Edge application for live anomaly detection;
- [Analyze MyMachine /Condition](https://support.industry.siemens.com/cs/document/109779880/), which creates and tracks a mechanical fingerprint for machine tools;
- digital-twin and Industrial Copilot offerings for engineering, simulation and maintenance assistance.

These are substantial products and must be treated as serious baselines, not dismissed. ForgeWorld's proposed differentiation is the **specific combination** below:

1. **A learned action-conditioned world model**, not only a normality model or fixed fingerprint.
2. **A strict distinction between control action and context**, preventing causal claims from metadata correlations.
3. **A dual healthy/current architecture:** a certified healthy reference remains frozen while a separate shadow model adapts online.
4. **Cross-modal state estimation:** sensors, PLC events, controller traces, vision, audio and quality measurements share one predictive state.
5. **Cross-asset and cross-vendor transfer** through OPC UA, MTConnect and ISA-95-aligned semantic tokens.
6. **Hierarchical propagation modeling:** component → machine → station → line.
7. **Counterfactual evaluation:** estimate consequences of changing feed, speed, setpoints, inspection frequency or maintenance timing.
8. **Uncertainty-aware abstention and safe advisory deployment**, rather than uncontrolled online actuation.
9. **Foundation pretraining plus few-cycle adaptation**, targeting machines, tools and recipes not represented during training.
10. **Benchmarkable research artifacts:** public-dataset protocols, leakage audits and reproducible comparisons against simple and strong baselines.

The moat is therefore not “we use AI for maintenance.” It is:

> **a vendor-neutral predictive state layer that learns plant dynamics, adapts without erasing the healthy reference, and links machine degradation to downstream quality and throughput.**

---

## 4. Research hypothesis

ForgeWorld tests five hypotheses.

### H1 — Predictive latent states outperform static anomaly embeddings under hard shifts

A state learned by predicting future multimodal tokens should preserve degradation and process dynamics better than a representation trained only for reconstruction or nearest-neighbor normality.

### H2 — Verified actions add causal and counterfactual value

Given strict action/context separation, including real control commands should improve future-state prediction and enable useful intervention ranking beyond metadata-only and sensor-only models.

### H3 — Dual adaptation detects degradation without normalizing it away

A machine-specific adaptive model should improve calibration under benign domain shifts, while its divergence from a frozen healthy model should retain sensitivity to progressive degradation.

### H4 — Cross-modal prediction provides earlier warning than any single modality

For example:

\[
\text{controller tracking error}
\rightarrow
\text{vibration change}
\rightarrow
\text{surface anomaly}
\rightarrow
\text{dimensional rejection}.
\]

Predicting the chain jointly should improve lead time and root-cause localization.

### H5 — A graph world model can identify propagation across stations

A learned station graph should separate a local inspection failure from an upstream process deviation and predict blocking, starvation, microstops and downstream defect risk.

---

## 5. What is genuinely new

The proposed system is named **ACME-JEPA** internally:

> **Adaptive Causal Multimodal Event-JEPA**.

It combines several ideas that exist separately but are not yet established as one industrial architecture.

### 5.1 Event-synchronous, multi-rate tokenization

Industrial data is heterogeneous:

- vibration may be sampled at 25–100 kHz;
- controller traces at 1–10 kHz;
- PLC tags at 10–1000 ms;
- quality measurements once per part;
- maintenance events once per day or month;
- images once per station or cycle.

Naively resampling everything to one rate either destroys high-frequency information or creates enormous sparse sequences. ForgeWorld uses hierarchical tokens:

\[
\text{raw samples}
\rightarrow
\text{local patches}
\rightarrow
\text{operation tokens}
\rightarrow
\text{cycle tokens}
\rightarrow
\text{batch/line tokens}.
\]

Every token retains:

```text
asset_id
component_id
signal_semantic_id
physical_unit
sampling_interval
event_time
operation_id
cycle_id
part_id
recipe_id
quality_link
confidence/missingness
```

### 5.2 Hybrid latent state: continuous dynamics plus discrete PLC state

A purely continuous Transformer is poorly matched to PLC sequences. ACME-JEPA maintains:

- continuous latent tokens for physical dynamics;
- discrete mode tokens for PLC/SFC states, alarms and transitions;
- object/part tokens for workpieces and tools;
- graph tokens for station and material-flow relationships.

The transition model is:

\[
(z_{t+1},m_{t+1},G_{t+1})
=
F_\theta(z_t,m_t,G_t,u_t,c_t),
\]

where \(m_t\) is the discrete operational mode.

### 5.3 Dual-timescale healthy reference and adaptive shadow

AdaJEPA adapts a world model using observed transitions. In industry, unrestricted adaptation is dangerous because gradual degradation can become the new normal. ForgeWorld therefore keeps:

- \(F_{\text{healthy}}\): frozen, versioned and tied to a certified healthy period;
- \(F_{\text{shadow},t}\): adapted to benign machine-, tool- and environment-specific shifts;
- \(F_{\text{fleet}}\): shared foundation model;
- adapters for asset, tool, recipe and session.

Key health signals include:

\[
D^{\text{pred}}_t
= d(F_{\text{healthy}}(s_t,u_t),s_{t+1}),
\]

\[
D^{\text{shadow}}_t
= d(F_{\text{shadow},t}(s_t,u_t),s_{t+1}),
\]

\[
D^{\text{drift}}_t
= d(F_{\text{healthy}}(s_t,u_t),F_{\text{shadow},t}(s_t,u_t)).
\]

Interpretation:

- both errors high: unseen or poorly modeled regime;
- healthy high, shadow low, stable adapter: benign asset-specific shift;
- healthy high and adapter drifts monotonically: likely degradation;
- shadow error rises too: abrupt anomaly or new fault.

### 5.4 Quality-linked world modeling

Most predictive-maintenance benchmarks stop at component state. ForgeWorld links each cycle to the part and its downstream quality measurements:

\[
p(q_{i,t+k}\mid z_t,u_{t:t+k},c_t,G_t).
\]

This enables a commercially meaningful target:

> **predict the first part that will leave tolerance, not only the date on which a component may fail.**

### 5.5 Counterfactual advisory planning

Given candidate interventions \(a^{(j)}\), the model estimates:

\[
\mathbb{E}[
C_{\text{quality}}
+C_{\text{downtime}}
+C_{\text{energy}}
+C_{\text{maintenance}}
\mid a^{(j)}].
\]

Initial interventions are advisory and bounded:

- inspect now versus after \(N\) cycles;
- change tool now versus later;
- reduce feed override within an approved envelope;
- change inspection sampling frequency;
- reduce line rate temporarily;
- route a batch to additional metrology.

ForgeWorld must not bypass PLC safety logic or certified controls.

### 5.6 Generative branch only where it adds value

MIRA demonstrates stable action-conditioned generative rollouts using representation autoencoders, diffusion forcing and multi-agent conditioning. ForgeWorld borrows these ideas selectively:

- latent flow/diffusion for multimodal future distributions;
- representation autoencoders for video, thermal and surface images;
- diffusion forcing for robustness to self-generated context;
- multi-entity conditioning for stations and workpieces.

The core planner and detector operate in latent/state space. Full pixel generation is optional because it is expensive and often unnecessary for maintenance.

---

## 6. System architecture

```text
                    ┌──────────────────────────────────────────────┐
                    │      Industrial semantic data contract       │
                    │ OPC UA / MTConnect / ISA-95 / custom mapping │
                    └──────────────────────┬───────────────────────┘
                                           │
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─▼────────┐ ┌──────────┐
  │ Sensors  │ │ CNC/PLC  │ │ Vision   │ │ Quality  │ │ CMMS/MES │
  │ raw/FFT  │ │ commands │ │ RGB/3D/T │ │ metrology│ │ events   │
  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │            │            │
  ┌────▼────┐  ┌────▼────┐  ┌────▼────────┐  │       ┌────▼────┐
  │Sensor   │  │Event /  │  │DINOv3 or   │  │       │Text/event│
  │tokenizer│  │PLC model│  │V-JEPA 2.1  │  │       │encoder   │
  └────┬────┘  └────┬────┘  └────┬────────┘  │       └────┬────┘
       └─────────────┴─────────────┴───────────┴────────────┘
                                  │
                         ┌────────▼────────┐
                         │ Multimodal JEPA │
                         │ state estimator │
                         └────────┬────────┘
                                  │ z_t, m_t, G_t
            ┌─────────────────────┼────────────────────────┐
            │                     │                        │
   ┌────────▼────────┐   ┌────────▼─────────┐    ┌────────▼────────┐
   │ Frozen healthy │   │ Adaptive shadow │    │ Probabilistic / │
   │ world model    │   │ AdaJEPA adapters│    │ MIRA-like branch│
   └────────┬────────┘   └────────┬─────────┘    └────────┬────────┘
            └─────────────────────┼────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │ Multi-horizon predictions│
                     │ sensors, modes, quality, │
                     │ failure, RUL, throughput │
                     └────────────┬────────────┘
                                  │
             ┌────────────────────┼─────────────────────┐
             │                    │                     │
      ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
      │ Early alert│      │ Root cause /│      │ Counterfactual│
      │ & lead time│      │ propagation │      │ advisory MPC  │
      └─────────────┘      └─────────────┘      └──────────────┘
```

---

## 7. Model components

### 7.1 Sensor encoder

The sensor path must support raw waveforms and engineered features.

Candidate backbones:

- 1D CNN / TCN as strong low-compute baselines;
- PatchTST/iTransformer/ModernTCN-style temporal models;
- masked temporal JEPA with span masking and latent re-masking;
- MOMENT, Chronos, TimesFM and Moirai representations as transfer baselines;
- time-frequency tokens from STFT, wavelet and order tracking;
- sparse mixture-of-experts by signal family and operating regime.

The primary representation is not trained only to reconstruct values. It predicts target spans, future tokens and cross-modal consequences.

### 7.2 Vision encoder

Priority:

1. DINOv3 frozen dense features;
2. V-JEPA 2.1 dense spatiotemporal features for video;
3. DINOv2, ResNet and EfficientAD as deployment baselines;
4. optional industrial continual pretraining;
5. representation autoencoder decoder for interpretable future-frame generation.

DINOv3 and V-JEPA 2.1 are complementary:

- DINOv3 is a strong image and dense-feature backbone;
- V-JEPA 2.1 explicitly adds temporal consistency and dense predictive supervision.

### 7.3 PLC/event encoder

The event model treats each command and transition as a structured token:

```text
COMMAND valve_04 OPEN setpoint=1 source=PLC timestamp=...
OBSERVE limit_switch_04 ON latency_ms=...
MODE station_2 CLAMPING -> MACHINING
ALARM drive_1 code=...
```

It predicts:

- next valid state;
- transition latency distribution;
- absent or duplicated transitions;
- impossible combinations;
- expected continuous response after a command.

### 7.4 Asset and line graph

Nodes:

- component;
- axis;
- spindle/motor;
- tool;
- machine;
- station;
- buffer;
- inspection point;
- product/part instance.

Edges:

- mechanical coupling;
- control dependency;
- material flow;
- shared utility;
- quality dependency;
- temporal precedence.

A graph Transformer or message-passing network propagates predicted effects between nodes.

### 7.5 Deterministic JEPA predictor

Used for compact, stable and inexpensive prediction:

\[
\hat z_{t+h}=F_\theta(z_{\le t},u_{t:t+h},c_t,G_t,h).
\]

### 7.6 Probabilistic predictor

Industrial futures are not always deterministic. The probabilistic path may use:

- mixture-density latent prediction;
- variational JEPA;
- conditional flow matching;
- diffusion forcing;
- ensemble posterior sampling.

It produces calibrated intervals and supports abstention.

### 7.7 Output heads

The project must support:

- future sensor forecasting;
- PLC transition and latency prediction;
- quality regression/classification;
- wear state and continuous wear estimate;
- failure-soon probability;
- survival/RUL distribution;
- anomaly localization by sensor, token, component and station;
- root-cause ranking;
- throughput/OEE and queue forecast;
- counterfactual intervention ranking.

---

## 8. Training objectives

The full loss is modular:

\[
\mathcal L=
\lambda_J\mathcal L_{\text{JEPA}}
+\lambda_F\mathcal L_{\text{forecast}}
+\lambda_X\mathcal L_{\text{cross-modal}}
+\lambda_Q\mathcal L_{\text{quality}}
+\lambda_S\mathcal L_{\text{survival}}
+\lambda_M\mathcal L_{\text{mode}}
+\lambda_G\mathcal L_{\text{graph}}
+\lambda_U\mathcal L_{\text{uncertainty}}
+\lambda_R\mathcal L_{\text{representation}}.
\]

### JEPA latent prediction

\[
\mathcal L_{\text{JEPA}}
=
\|P(E(x_{\text{context}}),u,c)-\operatorname{sg}(E(x_{\text{target}}))\|_2^2.
\]

### Multi-horizon forecasting

\[
\mathcal L_{\text{forecast}}
=
\sum_{h\in\mathcal H}w_h\,\ell(\hat y_{t+h},y_{t+h}).
\]

### Cross-modal prediction

Examples:

- sensor → expected surface representation;
- PLC command → response waveform;
- image/quality → expected sensor regime;
- upstream station → downstream quality.

### Survival/RUL

Use censored survival losses rather than fabricating exact failure times when maintenance occurs before failure.

### Physical consistency

Where valid:

- power ≈ torque × angular velocity;
- conservation/balance residuals;
- monotonic wear priors over appropriate intervals;
- known PLC transition constraints;
- actuator and setpoint bounds.

Physics terms are soft constraints unless the relationship is exact and calibrated.

### Anti-collapse and representation geometry

Evaluate:

- EMA target encoder versus LeJEPA/SIGReg;
- variance/covariance diagnostics;
- effective rank;
- token diversity;
- nearest-neighbor retrieval;
- downstream frozen probes.

---

## 9. Data contract: actions are not context

This is a non-negotiable design rule.

### `action`

A time-aligned command that can in principle be intervened upon:

- feed/RPM setpoint change;
- override command;
- axis target;
- valve/pump/drive command;
- robot trajectory command;
- recipe setpoint issued at time \(t\).

### `context`

A condition that explains the regime but is not treated as the immediate action:

- material or hardness;
- product family;
- tool identity;
- machine identity;
- shift/operator;
- ambient temperature;
- maintenance age;
- current recipe identifier.

### `outcome`

Must never be used as an input at or before the prediction time:

- future wear;
- cycle-to-failure;
- final pass/fail;
- future maintenance intervention;
- downstream quality measured after the prediction timestamp.

Every dataset adapter must produce a machine-readable leakage report.

---

## 10. Datasets

No public dataset contains the entire target problem. The project uses a curriculum: component dynamics, CNC degradation, PLC processes, visual quality, acoustic faults and line-level outcomes.

### 10.1 Priority A — must download and support first

| Dataset | Modality | Primary use | Access |
|---|---|---|---|
| NASA Milling Wear | force, vibration/acoustic/process, measured insert wear | CNC wear representation and transfer | [NASA Open Data](https://data.nasa.gov/dataset/milling-wear) |
| PHM 2010 Milling Challenge | dynamometer, accelerometer, acoustic emission, wear per cut | tool wear and RUL; leave-one-cutter-out | [PHM Society](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/) |
| CNC Mill Tool Wear / 18 runs | controller and process time series, tool condition | action/context audit and cycle segmentation | commonly mirrored on Kaggle; retain provenance and license in manifest |
| MetroPT-3 | pressure, temperature, current, valve states, real maintenance reports | real temporal anomaly, lead time and adaptation | [Scientific Data](https://www.nature.com/articles/s41597-022-01877-3), [Zenodo DOI in paper](https://doi.org/10.5281/zenodo.6854240) |
| Hydraulic System Condition Monitoring | pressure, flow, temperature, vibration; graded component states | multi-component degradation and severity | [UCI ID 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| SWaT | PLC/SCADA sensors and actuators, normal and attack scenarios | hybrid state/action transition modeling | [iTrust request page](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/) |
| Tennessee Eastman Process | process-control simulation | known interventions, process faults and counterfactuals | use an established implementation and record exact generator/version |
| MVTec AD 2 | high-resolution industrial images, difficult domain shifts | current visual benchmark; held-out evaluation server | [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) |
| MVTec LOCO AD | structural and logical anomalies | missing/wrong component and assembly logic | [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) |
| MVTec 3D-AD | aligned RGB and 3D scans | geometric defects and multimodal vision | [MVTec 3D-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad) |
| VisA | 10,821 industrial images with image/pixel labels | cross-dataset visual generalization | [AWS Open Data](https://registry.opendata.aws/visa/) |
| MIMII / DCASE machine sound | machine audio normal/anomalous | acoustic encoder and domain shift | [MIMII](https://zenodo.org/records/3384388), DCASE Task 2 releases |

### 10.2 Priority B — component and cross-domain generalization

| Dataset | Use | Access / note |
|---|---|---|
| CWRU Bearing Data Center | bearing fault baseline; strict load and drive-end split | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) |
| Paderborn Bearing Data Center | real and artificial bearing damage across conditions | [Paderborn download](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download) |
| IMS Bearing | run-to-failure bearing degradation | NASA/PHM repository or verified mirror; record checksum |
| XJTU-SY Bearing | accelerated run-to-failure and RUL | official university/research release when available |
| FEMTO-ST PRONOSTIA | bearing RUL under multiple conditions | PHM 2012 challenge source or verified mirror |
| AI4I 2020 | synthetic failure types and simple baseline sanity checks | [UCI ID 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| NASA C-MAPSS / N-CMAPSS | RUL protocol research | [NASA catalog](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data); official page reported unavailable at last review, so do not silently use an unverified mirror |
| WADI | larger water-distribution PLC/SCADA process | [iTrust datasets](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/) |
| HAI | industrial control system normal/anomalous operations | use official HAI release and versioned split |
| BATADAL | water-distribution attack/anomaly benchmark | official challenge release |
| Bosch Production Line Performance | part-level measurements across production routes | Kaggle competition; anonymized and unsuitable for causal claims |
| SECOM | semiconductor process features and pass/fail | [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) |

### 10.3 Priority C — visual breadth and logical/temporal inspection

| Dataset | Use | Access |
|---|---|---|
| MVTec AD | regression benchmark and compatibility | [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) |
| KolektorSDD / SDD2 | surface defect segmentation | official Kolektor releases |
| MPDD | metal part defect detection under pose variation | official paper/repository release |
| AeBAD | blade images/video with domain shift | [paper and dataset repository](https://arxiv.org/abs/2304.02216) |
| Real-IAD / Real-IAD Variety | large real industrial anomaly diversity | use official release, verify license and download availability |
| BTAD | industrial visual anomalies | official release |
| NEU surface defects | steel surface classification | Northeastern University release |
| DAGM 2007 | synthetic industrial textures | official competition release |
| Severstal Steel Defect | steel-strip segmentation | Kaggle; license and competition terms apply |

### 10.4 Public data is insufficient for the final claim

The decisive dataset must be collected from a real pilot and contain synchronized:

- raw and aggregated sensor streams;
- CNC/PLC commands and actual responses;
- exact operation/cycle/part identifiers;
- tool and material metadata;
- images or metrology;
- alarm and maintenance events;
- replaced component and technician diagnosis;
- censored survival information;
- normal recipe changes;
- controlled safe interventions where possible.

At minimum, the pilot should cover:

- two or more nominally similar machines;
- several tools/components run across their lifecycle;
- multiple materials/recipes;
- maintenance reset events;
- quality measurements linked to part IDs;
- at least one held-out machine or site.

### 10.5 Dataset storage layout

```text
data/
  raw/
    sensor/
    plc/
    visual/
    audio/
    quality/
  external/            # immutable extracted public datasets
  interim/             # synchronized but not model-ready
  processed/
    tokens/
    cycles/
    graphs/
  manifests/
    datasets.yaml
    licenses.yaml
    checksums.csv
    leakage_reports/
  splits/
    by_asset/
    by_tool/
    by_time/
    by_site/
```

Raw data is immutable. Every transformation is versioned.

---

## 11. Papers and technical reading plan

### 11.1 JEPA foundations

1. **A Path Towards Autonomous Machine Intelligence** — conceptual JEPA and world-model program.  
   https://openreview.net/forum?id=BZ5a1r-kVsf
2. **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** — masking, target encoder and semantic latent prediction.  
   https://arxiv.org/abs/2301.08243
3. **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video** — video-only feature prediction.  
   https://arxiv.org/abs/2404.08471
4. **V-JEPA 2** — internet-scale video pretraining and action-conditioned robot planning.  
   https://arxiv.org/abs/2506.09985
5. **V-JEPA 2.1** — dense predictive loss, deep self-supervision and stronger dense/temporal features.  
   https://arxiv.org/abs/2603.14482
6. **LeJEPA** — SIGReg and collapse-resistant joint-embedding training.  
   https://arxiv.org/abs/2511.08544
7. **JEPA-DNA** — useful for its dual masking/re-masking design, not for its genomic domain.  
   https://arxiv.org/abs/2602.17162

### 11.2 Visual representation backbones

1. **DINO** — emergent dense semantics.  
   https://arxiv.org/abs/2104.14294
2. **DINOv2** — robust self-supervised visual features.  
   https://arxiv.org/abs/2304.07193
3. **DINOv3** — scaling, Gram anchoring and state-of-the-art dense frozen features.  
   https://arxiv.org/abs/2508.10104
4. **DINO-WM** — offline action-conditioned world model over DINO patch features.  
   https://arxiv.org/abs/2411.04983

### 11.3 Adaptive and long-horizon world models

1. **AdaJEPA: An Adaptive Latent World Model** — self-supervised test-time adaptation inside MPC.  
   https://arxiv.org/abs/2606.32026
2. **FF-JEPA** — hierarchical latent subgoal planning for longer horizons.  
   https://arxiv.org/abs/2606.09311
3. **Variational JEPA / Var-JEPA** — uncertainty-aware latent prediction.  
   https://arxiv.org/abs/2601.14354  
   https://arxiv.org/abs/2603.20111
4. **TD-MPC2** — scalable decoder-free latent world models and planning.  
   https://arxiv.org/abs/2310.16828
5. **DreamerV3** — policy learning through imagined trajectories.  
   https://arxiv.org/abs/2301.04104
6. **MuZero** — planning with a learned model that predicts planning-relevant quantities.  
   https://arxiv.org/abs/1911.08265

### 11.4 Generative world models

1. **MIRA: Multiplayer Interactive World Models with Representation Autoencoders** — action attribution, multi-view coherence, real-time latent diffusion and targeted physical evaluations.  
   https://arxiv.org/abs/2607.05352
2. **Diffusion Forcing** — per-token noise levels and robust rollouts beyond the training horizon.  
   https://arxiv.org/abs/2407.01392
3. **Diffusion Transformers with Representation Autoencoders** — semantically rich pretrained encoder plus learned decoder.  
   https://arxiv.org/abs/2510.11690

### 11.5 Time-series foundation models

Use these as transfer and forecasting baselines, not as proof that a generic forecaster is an industrial world model.

1. **MOMENT** — multi-task open time-series foundation models.  
   https://arxiv.org/abs/2402.03885
2. **TimesFM** — decoder-only zero-shot forecasting.  
   https://arxiv.org/abs/2310.10688
3. **Moirai / Moirai-MoE** — universal forecasting and token-level expert specialization.  
   https://arxiv.org/abs/2402.02592  
   https://arxiv.org/abs/2410.10469
4. **Chronos / Chronos-2** — probabilistic tokenized forecasting and multivariate in-context forecasting.  
   https://arxiv.org/abs/2403.07815  
   https://arxiv.org/abs/2510.15821

### 11.6 Industrial anomaly and quality baselines

1. **PatchCore** — nominal patch memory bank.  
   https://arxiv.org/abs/2106.08265
2. **PaDiM** — patch distribution modeling.  
   https://arxiv.org/abs/2011.08785
3. **EfficientAD** — real-time student/teacher industrial anomaly detection.  
   https://arxiv.org/abs/2303.14535
4. **Anomaly Transformer** — association discrepancy for multivariate time series.  
   https://arxiv.org/abs/2110.02642
5. **TranAD** — Transformer anomaly detection and diagnosis.  
   https://arxiv.org/abs/2201.07284
6. **MVTec AD 2** — modern benchmark designed because earlier visual benchmarks were saturating.  
   https://arxiv.org/abs/2503.21622

### 11.7 Test-time adaptation and continual calibration

Read and benchmark:

- TENT: entropy minimization at test time;
- CoTTA: continual test-time adaptation;
- EATA: efficient anti-forgetting adaptation;
- SAR: sharpness-aware reliable adaptation;
- online normalization and conformal calibration;
- survival-model recalibration and drift detection.

AdaJEPA is the architectural starting point, but industrial adaptation must add frozen references, replay, trust regions, uncertainty gates and maintenance resets.

### 11.8 Physics, causality and hybrid systems

Research topics:

- Neural ODEs and controlled differential equations;
- universal differential equations;
- graph network simulators;
- differentiable state-space models;
- switching linear dynamical systems;
- neural hybrid automata;
- causal discovery with interventions;
- structural causal models for process variables;
- survival analysis with censoring;
- conformal prediction under covariate shift.

The project should not label correlation as causality unless the data contains interventions or the claim is backed by a justified causal design.

---

## 12. Repository design

```text
forgeworld/
  README.md
  AGENTS.md
  pyproject.toml
  configs/
    data/
    pretrain/
    world_model/
    adaptation/
    benchmarks/
    deployment/
  src/forgeworld/
    data/
      contracts.py
      synchronization.py
      resampling.py
      cycle_segmentation.py
      leakage.py
      datasets/
    semantics/
      opcua.py
      mtconnect.py
      isa95.py
      units.py
    tokenizers/
      sensor.py
      event.py
      vision.py
      audio.py
      quality.py
    encoders/
      sensor_jepa.py
      visual_backbones.py
      plc_encoder.py
      multimodal_fusion.py
    world_models/
      deterministic_jepa.py
      probabilistic_jepa.py
      graph_world_model.py
      generative_rae.py
    adaptation/
      healthy_reference.py
      shadow_adapter.py
      replay.py
      trust_region.py
    heads/
      forecasting.py
      anomaly.py
      quality.py
      survival.py
      root_cause.py
      throughput.py
    planning/
      cem.py
      bounded_mpc.py
      intervention_cost.py
    evaluation/
      metrics.py
      protocols.py
      calibration.py
      lead_time.py
      sota_gate.py
    deployment/
      edge_runtime.py
      shadow_mode.py
      opcua_connector.py
      audit_log.py
  scripts/
    download/
    prepare/
    train/
    evaluate/
    deploy/
  tests/
    unit/
    integration/
    leakage/
    reproducibility/
  data/
  outputs/
  reports/
```

---

## 13. Implementation roadmap

### Phase 0 — preserve and formalize the existing MVP

Deliverables:

- reproduce current Sensor-JEPA and visual baselines;
- freeze exact dataset manifests and splits;
- install official MiniROCKET/MultiROCKET where applicable;
- benchmark DINOv3, DINOv2, EfficientAD, PatchCore and PaDiM;
- document the existing result as a baseline, not the final system.

Go/no-go:

- all tests pass;
- results reproducible across three seeds;
- no forbidden lifecycle/outcome features.

### Phase 1 — Industrial semantic token layer

Deliverables:

- canonical signal schema with units and asset hierarchy;
- action/context/outcome classification;
- event-time synchronization;
- multi-rate patching;
- PLC transition extraction;
- cycle/part/quality linkage;
- dataset adapters for Priority A datasets.

Go/no-go:

- round-trip dataset validation;
- timestamp and unit audits;
- leakage reports generated automatically.

### Phase 2 — unimodal predictive foundations

Deliverables:

- Sensor-JEPA with raw and engineered input paths;
- DINOv3/V-JEPA 2.1 visual baselines;
- PLC next-state and latency model;
- acoustic encoder;
- multi-horizon prediction heads.

Required comparisons:

- raw features;
- engineered features;
- classical statistical models;
- ROCKET family;
- TCN/GRU/1D CNN;
- modern Transformer/time-series foundation models.

### Phase 3 — action-conditioned causal protocol

Deliverables:

- real command tokens on SWaT/TEP and pilot CNC/PLC data;
- action-only, context-only, no-action and shuffled-action ablations;
- counterfactual prediction tests;
- intervention consistency and sensitivity metrics.

A world-model claim is not allowed before this phase.

### Phase 4 — dual AdaJEPA adaptation

Deliverables:

- frozen healthy checkpoint;
- shadow adapters by asset/tool/recipe;
- small replay buffer;
- trust-region update;
- uncertainty gate;
- maintenance-reset protocol;
- comparison against no adaptation, full fine-tuning, TENT/CoTTA-style baselines and simple recalibration.

### Phase 5 — multimodal quality linkage

Deliverables:

- joint sensor/PLC/vision/quality state;
- cross-modal masked prediction;
- first-out-of-tolerance forecasting;
- source-to-defect attribution;
- missing-modality robustness.

### Phase 6 — hierarchical line world model

Deliverables:

- component/machine/station graph;
- material-flow tokens;
- propagation prediction;
- blocking/starvation and throughput forecasts;
- line-level counterfactuals.

### Phase 7 — industrial pilot in shadow mode

Deliverables:

- edge ingestion;
- no-write connection to PLC/CNC/MES/SCADA;
- alerts with confidence and provenance;
- operator feedback;
- lead-time and false-alarm report;
- cyber-security and rollback design.

No automated parameter changes until shadow-mode acceptance criteria are met.

---

## 14. Benchmark protocols

### 14.1 Required split types

Random row splits are prohibited for lifecycle and process sequences.

Use:

- leave-one-tool-out;
- leave-one-machine-out;
- leave-one-operating-condition-out;
- chronological future split;
- leave-one-material/recipe-out;
- leave-one-site-out;
- pre-maintenance versus post-maintenance separation;
- visual train-normal/test-shift protocols.

### 14.2 Required baselines

#### Sensors

- persistence and seasonal-naive forecasts;
- linear/logistic regression;
- random forest, XGBoost/LightGBM where licensing permits;
- engineered vibration/current/process features;
- 1D CNN, TCN, GRU/LSTM;
- MiniROCKET/MultiROCKET;
- PatchTST/iTransformer/ModernTCN or current strong equivalents;
- MOMENT/Chronos/TimesFM/Moirai where task-compatible.

#### Vision

- DINOv3 and DINOv2 nearest-neighbor memory;
- PatchCore;
- PaDiM;
- EfficientAD;
- SimpleNet/FastFlow/Reverse Distillation when reproducible;
- official MVTec AD 2 evaluation.

#### World models

- current-state classifier;
- no-action predictor;
- context-conditioned predictor;
- shuffled-action predictor;
- one-step state-space model;
- DINO-WM/V-JEPA-style predictor where applicable;
- generative versus non-generative latent predictor;
- frozen versus adaptive versus dual adaptive.

### 14.3 Metrics

#### Predictive state

- latent cosine/MSE only as diagnostic;
- decoded signal MAE/RMSE/CRPS;
- multi-step degradation curves;
- calibration and interval coverage;
- mode-transition accuracy and latency error.

#### Anomaly/failure

- event-based precision/recall/F1;
- AUROC and AUPRC;
- false alarms per operating hour or 1,000 cycles;
- mean/median lead time;
- missed critical events;
- time-to-detection;
- NAB or range-based metrics where appropriate.

#### RUL/survival

- RMSE/MAE with caution;
- NASA scoring function where relevant;
- concordance index;
- integrated Brier score;
- calibration by horizon;
- censoring-aware likelihood.

#### Quality

- first-out-of-tolerance lead time;
- scrap/rework recall at fixed inspection budget;
- top-k inspection precision;
- dimensional error MAE;
- defect localization AU-PRO/pixel AUPRC for vision.

#### Adaptation

- improvement after 1, 5, 20 and 100 transitions;
- benign-shift calibration;
- degradation retention;
- catastrophic forgetting;
- adapter drift magnitude;
- compute and memory per update.

#### Business

- avoided unplanned downtime;
- avoided scrap/rework;
- unnecessary inspections;
- maintenance lead time;
- operator acceptance;
- inference cost per asset.

---

## 15. SOTA claim protocol

The project may use the phrases **research target**, **candidate**, or **competitive** before this gate. It may claim SOTA only when all relevant conditions hold.

### Dataset and protocol

- public benchmark with official or widely accepted split;
- no train/test leakage by tool, asset, time, part or lifecycle;
- thresholds selected using training/validation only;
- test labels never used to choose score direction or hyperparameters;
- exact data version and checksum published.

### Baselines

- official or faithful implementations of the strongest published baselines;
- same input modalities and supervision budget;
- same split and metrics;
- simple baselines included;
- no “lite” substitute presented as equivalent to the published method.

### Statistics

- at least five seeds for small/medium datasets, or justified deterministic repeated splits;
- mean, standard deviation and confidence intervals;
- paired significance or bootstrap test where appropriate;
- per-asset/per-category results, not only pooled averages.

### Generalization

At least two of:

- held-out machine;
- held-out tool/component;
- held-out material/recipe;
- held-out site;
- cross-dataset evaluation;
- real pilot after public-data development.

### Operational validity

- false alarms and lead time reported;
- inference and adaptation latency measured on target hardware;
- uncertainty calibrated;
- failure cases documented;
- shadow-mode field evaluation completed.

### Claim wording

Even after a benchmark SOTA result, distinguish:

- “SOTA on dataset X under protocol Y”;
- “best public result we could reproduce”;
- “production value demonstrated in pilot Z.”

Never infer plant-wide superiority from one public benchmark.

---

## 16. Initial experiments that can falsify the idea

A serious research project must define failure conditions.

### Experiment A — Does future latent prediction add information?

Compare under leave-one-tool-out:

- metadata only;
- engineered sensors;
- current latent;
- predicted future latent;
- engineered + latent;
- engineered + predicted future.

Fail condition: no consistent incremental value over engineered features across seeds and tools.

### Experiment B — Do actions matter?

On TEP/SWaT or real commands:

- correct action sequence;
- shuffled actions;
- context only;
- no actions.

Fail condition: shuffled actions perform equally, indicating that the model ignores actions or exploits context.

### Experiment C — Does dual adaptation preserve degradation?

Simulate/observe:

- benign machine offset;
- recipe shift;
- progressive wear;
- abrupt fault.

Compare:

- frozen;
- unrestricted online adaptation;
- dual healthy/shadow adaptation.

Fail condition: the dual model does not improve benign calibration or still absorbs progressive faults.

### Experiment D — Does multimodal fusion improve lead time?

Compare sensor, PLC, vision/quality and fused models at fixed false-alarm rate.

Fail condition: fused model adds no lead time or degrades robustness when a modality is missing.

### Experiment E — Can the line graph locate propagation?

Inject or use known upstream deviations and evaluate root-cause station ranking.

Fail condition: graph model does not outperform temporal correlation and simple lag analysis.

---

## 17. Deployment principles

1. **Shadow mode first.** Read-only integration before recommendations.
2. **No safety replacement.** Certified PLC/CNC safety remains authoritative.
3. **Bounded recommendations.** Any setpoint advice must stay within approved envelopes.
4. **Human review.** High-cost maintenance actions require confirmation.
5. **Provenance.** Every alert stores model version, data window, contributing signals and uncertainty.
6. **Graceful degradation.** Missing sensors or connectivity must not produce confident guesses.
7. **Edge-first option.** Sensitive plant data can remain on premises.
8. **Rollback.** Frozen model and prior adapter are always recoverable.
9. **Maintenance reset.** Replacement/calibration events create explicit state boundaries.
10. **Cybersecurity.** Treat data connectors and model update channels as industrial attack surfaces.

---

## 18. Proposed command-line interface

These commands describe the target interface; implementation is incremental.

```bash
# Inventory and licensing
python scripts/data/build_manifest.py --root data/raw
python scripts/data/audit_licenses.py
python scripts/data/audit_leakage.py --config configs/data/cnc.yaml

# Public data
python scripts/download/download_nasa_milling.py
python scripts/download/download_phm2010.py
python scripts/download/download_metropt3.py
python scripts/download/requested_dataset_instructions.py swat

# Pretraining
python scripts/train/pretrain_sensor_jepa.py --config configs/pretrain/sensor_foundation.yaml
python scripts/train/extract_dinov3_features.py --config configs/pretrain/visual_foundation.yaml
python scripts/train/pretrain_plc_model.py --config configs/pretrain/plc_swat.yaml

# World model
python scripts/train/train_world_model.py --config configs/world_model/cnc_action_conditioned.yaml
python scripts/train/train_multimodal_world_model.py --config configs/world_model/cnc_quality.yaml

# Adaptation
python scripts/evaluate/eval_dual_adaptation.py --config configs/adaptation/metropt3.yaml

# SOTA gate
python scripts/evaluate/run_benchmark_suite.py --config configs/benchmarks/public_suite.yaml
python scripts/evaluate/check_sota_gate.py outputs/benchmark_suite

# Pilot
python scripts/deploy/run_shadow_mode.py --config configs/deployment/site_a.yaml
```

---

## 19. Near-term product

The first product should not promise a universal factory model. It should deliver one measurable outcome:

> **Predict the next quality-threatening degradation event in a repetitive CNC or PLC-controlled process with an actionable lead time and a controlled false-alarm rate.**

### Pilot inputs

- 4–20 high-value signals;
- commands/setpoints and modes;
- operation/cycle/part IDs;
- maintenance log;
- one quality measurement or pass/fail label;
- at least several months or multiple component lifecycles.

### Pilot outputs

- current health state;
- risk in next \(N\) cycles/hours;
- expected first quality deviation;
- top contributing operations/signals;
- likely origin component/station;
- uncertainty;
- recommended inspection window;
- model-versus-baseline evidence.

### Acceptance target

Targets must be agreed per plant. Example gates:

- materially greater lead time than current thresholds;
- fewer false alarms at equal recall;
- lower scrap/rework under a simulated or shadow inspection policy;
- stable performance across at least one held-out recipe or machine;
- operator explanation judged useful.

---

## 20. What this project will not claim

ForgeWorld will not claim:

- a universal model of every factory;
- causal control from datasets that contain only context;
- exact RUL when the dataset lacks run-to-failure or censoring information;
- production readiness from public datasets;
- SOTA from one seed or an internal split;
- that a JEPA must outperform engineered features;
- that visual generation is necessary for every use case;
- that online adaptation is safe without a frozen reference and update guardrails.

The research is successful even if it discovers that a simpler model is superior for a given plant. The deliverable is reliable decision support, not architectural ideology.

---

## 21. Relationship to the existing Industrial JEPA MVP

The current MVP already provides valuable building blocks:

- Sensor-JEPA pretraining;
- token-level future prediction;
- latent surprise features;
- engineered-feature comparisons;
- DINOv2/ResNet + PatchCore/PaDiM visual baselines;
- leakage audits;
- hierarchical score aggregation;
- reproducible scripts and reports.

ForgeWorld changes the research question from:

> “Does this window or image indicate risk?”

into:

> “What hidden process state generated these observations, how will it evolve under each command, how will the deviation affect product and line outcomes, and which safe intervention changes that future?”

The existing MVP is Phase 0, not discarded work.

---

## 22. License and dataset compliance

Code license should be chosen explicitly before public release. Dataset licenses differ and may prohibit commercial use or redistribution.

Rules:

- download scripts retrieve data from the owner; do not commit datasets;
- store source URL, terms, version, checksum and access date;
- restricted/request-only datasets require manual acceptance;
- do not redistribute MVTec or other datasets contrary to their terms;
- customer data never enters public artifacts;
- model weights trained on restricted data require a separate legal review.

---

## 23. Final vision

ForgeWorld is not intended to replace CNCs, PLCs, digital twins or maintenance systems. It is a predictive intelligence layer above them.

The end state is a model that can say:

> “Under the current recipe, this machine should have produced this force/current/vibration and this downstream quality distribution. The observed deviation began during operation OP20, is inconsistent with material and ambient changes, and is propagating from spindle/tool behavior into surface quality. The healthy reference predicts tolerance failure within 40–70 pieces. The adaptive shadow confirms a machine-specific shift but its drift is monotonic, so it should not be normalized away. Inspecting or changing tool T07 now has lower expected cost than continuing; reducing feed by 8% is predicted to extend the safe window but does not remove the underlying degradation.”

Achieving that outcome—reliably, with calibrated uncertainty and strict validation—is the SOTA target.
