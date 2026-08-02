# PROJECT_CONTEXT.md
> Paste this whole file into any AI coding assistant (Claude, ChatGPT, Copilot Chat) before
> asking it for help on this repo. It gives full project context in one shot so you don't have
> to re-explain the goal every time. Keep this file updated as the project evolves — if it goes
> stale, everyone's AI-assisted coding gets worse.

## 1. What this project is

We're building a multimodal deep learning model to predict **sepsis onset** (not mortality)
ahead of time, using three data sources from MIMIC-IV / MIMIC-CXR / MIMIC-IV-Note:
- Irregular time-series (vitals + labs)
- Clinical notes (radiology reports + discharge notes)
- Chest X-rays (CXR)

## 2. The gap we're solving (read this before touching model code)

Every existing multimodal fusion architecture decides how much to trust a piece of data
based on **whether it exists** (present/absent — e.g. FuseMoE, DrFuse) or **what it contains**
(content-based confidence — e.g. MedPatch). None of them decide based on **how old it is
relative to how fast the patient is currently changing**. A CXR from 30 hours ago means
something very different for a stable patient than for one who's been deteriorating for the
last 4 hours — but every current model treats those two cases identically.

## 3. Our solution: two named components

- **SDCA (Staleness-Decay Cross-Attention)**: a learned "trust dial" applied to every token
  before fusion, computed from (elapsed time since the observation, current patient trajectory
  volatility). Old data on an unstable patient gets down-weighted more than old data on a
  stable patient.
- **SARL (Staleness-Aware Regularization Loss)**: an auxiliary training term that keeps the
  decay function from collapsing to something degenerate (e.g. ignoring time entirely, or
  being overconfident when only stale data is available).

Both sit on top of **our own fusion backbone** — a standard multi-head cross-attention
design across the three modality token streams, built and owned by us. We do NOT fork
another paper's fusion architecture (MedPatch/FuseMoE/DrFuse) — we build our own, prove
it beats all baselines, then ablate SDCA/SARL against our own architecture. See
`docs/gap_and_solution.md` for the full explanation if you need more depth.

## 4. Closest related work (know these before implementing anything)

| Paper | What it does | What it's missing (our gap) |
|---|---|---|
| MedFuse | LSTM fusion, TS+CXR only | No notes, no staleness modeling |
| FuseMoE | mTAND + sparse MoE, TS+notes+CXR/ECG | Missingness handled as binary routing, no staleness |
| DrFuse | Disentangled shared/unique latent space, TS+CXR only | Doesn't scale past 2 modalities, no staleness |
| MedPatch | Token-level content confidence (calibration-based), TS+CXR+notes | Uses only most-recent CXR snapshot; confidence is static, not time-aware. **Their own paper names multi-timepoint integration as future work — this is exactly our contribution.** |
| DDL-CXR | Generates a new synthetic CXR via diffusion to fix staleness | CXR-only, expensive, can't generalize to notes |

## 5. Task definition (locked — do not change without team discussion)

**Cohort construction (filtering, applied BEFORE labeling):**
- One ICU stay per patient — first eligible stay only, not all admissions
- Adults only (age >= 18)
- Minimum ICU length of stay (exclude very short stays, e.g. < 4-6h — not enough
  time for a meaningful lead-time prediction task)
- Exclude admissions where sepsis criteria are already met at/before ICU admission
  (no "early prediction" possible for these — exclude, don't label positive)
- Exclude admissions with insufficient chart/lab density to reliably compute SOFA
- Split assignment is at the subject_id level, not hadm_id level — if a patient
  somehow contributes more than one admission, ALL of that patient's admissions
  go in the same split (never split across train/val/test for the same patient)

**Task formulation (how predictions are actually made — locked, do not change without
team discussion):**
- **NOT** a fixed-window snapshot task (unlike MedFuse/FuseMoE/MedPatch's 48-IHM style).
  This is a continuous, rolling early-warning task, following the standard framing used
  by the PhysioNet 2019 Sepsis Challenge and Moor et al. (2019)'s GP-TCN paper.
- A prediction timepoint is generated every hour per admission, starting after a short
  observation/buffer period, stopping at sepsis onset or discharge for positive patients
  (no timepoints generated after onset) -- following the exact setup used by SepsisCalc
  and SepsisLab (both run hourly sliding predictions until diagnosis or discharge).
- Single fixed prediction horizon W = 4 hours (matches SepsisCalc/SepsisLab directly,
  a stronger and more specific precedent than the general PhysioNet Challenge
  convention): label = 1 at timepoint t if onset occurs within the next 4h
  (onset_time - t <= 4h), label = 0 otherwise. Negative (non-septic) patients are
  label = 0 at every valid hourly timepoint across their stay.
- ONE model is trained on this single task -- NOT four separately trained lead-time
  models.
- The 2h/4h/6h/12h lead-time sweep (Table 2 / Figure 3) is produced by stratifying test
  predictions by TRUE distance-to-onset after training, not by training separate models.
- This hourly expansion happens dynamically in the dataset/dataloader code at training
  time (experiments/), NOT precomputed into sepsis_labels.parquet -- that file stays
  admission-level as already defined in docs/data_schema.md.
- Negative-timepoint subsampling during training is expected given severe class
  imbalance at the timepoint level (worse than the patient-level positive rate) --
  keep all positive timepoints, downsample negatives to a documented ratio.


- Label: Sepsis-3 (suspicion of infection + SOFA score increase ≥2 within a −48h/+24h window)
- Prediction task: early warning at lead times of **2h / 4h / 6h / 12h before onset**
- Primary metrics: AUROC, AUPRC (AUPRC is primary given class imbalance), ECE for calibration
- See `preprocessing/label_sepsis3.py` for the single source of truth on both cohort
  construction and labeling — never reimplement this logic elsewhere.

## 6. Repo map (what lives where)

```
sepsis-repo/
├── PROJECT_CONTEXT.md          <- this file
├── docs/
│   ├── gap_and_solution.md     <- full explanation of the gap, for deeper context
│   └── experiment_blueprint.md <- full table/figure/ablation spec for the paper
├── data/
│   ├── cohort/                 <- Sepsis-3 cohort construction SQL/scripts
│   ├── raw_links/               <- scripts pointing to physionet paths, NOT raw data
│   └── splits/                  <- fixed train/val/test patient ID lists, versioned once
├── preprocessing/
│   ├── ehr_extraction.py
│   ├── notes_extraction.py
│   ├── cxr_linking.py           <- keep FULL timestamped sequence, not just latest
│   └── label_sepsis3.py         <- single source of truth, locked
├── models/
│   ├── encoders/                <- TS/notes/CXR encoders, standard/off-the-shelf on purpose
│   ├── baselines/                <- MedFuse, FuseMoE, MedPatch, DrFuse reproductions
│   └── ours/
│       ├── backbone.py           <- our own fusion architecture
│       ├── sdca.py               <- Staleness-Decay Cross-Attention
│       └── sarl.py               <- Staleness-Aware Regularization Loss
├── experiments/
│   ├── configs/                  <- one YAML per run: task, lead-time, modality combo, seed
│   ├── train.py
│   ├── evaluate.py               <- AUROC/AUPRC/ECE, bootstrapped CIs
│   └── stress_tests/             <- missingness sweep, staleness-injection sweep
├── notebooks/                     <- exploratory only, nothing load-bearing lives here
└── results/                       <- versioned CSVs/JSON of metrics per run, no model weights
```

## 7. Current status
> Update this section every week so everyone (and every AI assistant) knows what's done.

- [ ] Cohort construction / Sepsis-3 labeling
- [ ] EHR/notes/CXR preprocessing pipelines
- [ ] Baseline reproductions (MedFuse / FuseMoE / MedPatch / DrFuse)
- [ ] Our backbone architecture
- [ ] SDCA implementation
- [ ] SARL implementation
- [ ] Main results (Table 2 / Figure 3)
- [ ] Ablations (Table 3)
- [ ] Stress tests (Table 4 / Figure 4)

## 8. Rules that must never be broken
1. Every baseline number reported must come from **our own re-run** on our cohort — never
   copy numbers from the original papers.
2. `preprocessing/label_sepsis3.py` is the single source of truth for sepsis labels — if you
   need a different definition, discuss with the team first, don't fork the logic silently.
3. Ablations (Table 3) are performed against **our own architecture**, not a baseline's.
4. Don't merge to `main` without a reviewed PR — see `CONTRIBUTING.md`.
5. **Every model — every baseline reproduction AND our own architecture — consumes data
   from the SAME `preprocessing/` pipeline.** Never let a baseline bring its own separate
   extraction/normalization code (e.g. MedPatch's `mimic4extract/`, FuseMoE's own scripts).
   If you find yourself copying a baseline's preprocessing wholesale instead of adapting its
   logic into our shared pipeline, stop — this breaks the fairness of every comparison in
   Table 2. It's fine (encouraged) to reference a baseline's variable set, normalization, or
   split-alignment conventions when building our shared pipeline — just build it once, here,
   for everyone.