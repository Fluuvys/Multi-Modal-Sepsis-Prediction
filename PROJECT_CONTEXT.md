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

- Label: Sepsis-3 (suspicion of infection + SOFA score increase ≥2 within a −48h/+24h window)
- Prediction task: early warning at lead times of **2h / 4h / 6h / 12h before onset**
- Primary metrics: AUROC, AUPRC (AUPRC is primary given class imbalance), ECE for calibration
- See `preprocessing/label_sepsis3.py` for the single source of truth on labeling — never
  reimplement this logic elsewhere.

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
