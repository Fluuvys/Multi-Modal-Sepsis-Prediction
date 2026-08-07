# PROJECT_CONTEXT.md
> Paste this whole file into any AI coding assistant (Claude, ChatGPT, Copilot Chat) before
> asking it for help on this repo. It gives full project context in one shot so you don't have
> to re-explain the goal every time. Keep this file updated as the project evolves — if it goes
> stale, everyone's AI-assisted coding gets worse. Last major revision: refined SDCA/SARL
> mechanism design, added the Step 0 signal-verification gate, added the diagnostic-suspicion-
> bias confound check.

## 1. What this project is

We're building a multimodal deep learning model to predict **sepsis onset** (not mortality)
ahead of time, using three data sources from MIMIC-IV / MIMIC-CXR / MIMIC-IV-Note:
- Irregular time-series (vitals + labs)
- Clinical notes (radiology reports + discharge notes)
- Chest X-rays (CXR)

## 2. The gap we're solving (read this before touching model code)

Every existing multimodal fusion architecture decides how much to trust a piece of data
based on **whether it exists** (present/absent — e.g. FuseMoE, DrFuse, MedFuse) or **what it
contains** (content-based confidence — e.g. MedPatch's calibrated token confidence). None of
them decide based on **how old it is relative to how fast the patient is currently changing**.
A CXR from 30 hours ago means something very different for a stable patient than for one
who's been deteriorating for the last 4 hours — but every current model treats those two
cases identically, because neither elapsed time nor patient volatility is a signal any current
fusion mechanism uses, jointly, to set trust.

The one architecture that engages with staleness at all, DDL-CXR, fixes it by generating a
brand-new synthetic CXR via latent diffusion — expensive, and structurally CXR-only (can't
generate a new clinical note the same way). So: missingness is solved (binary), content
confidence is solved, and staleness is solved only for images, only through expensive
generation. There is no general, lightweight, modality-agnostic mechanism where fusion
weight is parameterized jointly by elapsed time AND patient volatility.

See `docs/gap_and_solution.md` for the full explanation with worked examples.

## 3. Signal verification (REQUIRED gate — do this before writing any model code)

Before building anything, verify the phenomenon actually exists in the data. This is a data
analysis question, not a modeling question, and it determines whether the whole project has
something real to learn from.

**Step 0 diagnostic (Milestone 2.5, blocks Milestone 3):**
1. Train/use a plain baseline with NO staleness-awareness (carry-forward CXR, naive fusion).
2. For every prediction, log: Δt (age of last CXR/note at prediction time), a volatility
   proxy (rolling std/derivative of a shock index or vitals over trailing hours), and the
   model's error or calibration (is it overconfident specifically when data is old AND the
   patient is volatile?).
3. Regress error/overconfidence against Δt, volatility, and their INTERACTION term
   (Δt × volatility). We need the interaction to matter beyond the two main effects alone —
   that's the specific signature a decay mechanism would have something real to learn from.
4. Crosstab Δt (binned) × volatility (binned), check cell counts. Volatile patients often get
   RE-IMAGED precisely because they're volatile (clinicians order a fresh CXR when things go
   wrong) — so the "stale + volatile" cell may be sparse in MIMIC-IV. If it's too sparse,
   there's little for a learned gate to train on.

**If the interaction term isn't there, or the stale+volatile cell is too sparse, STOP and
re-evaluate before building SDCA/SARL** — a gating mechanism, however elegant, can't learn
something that isn't in the data. This diagnostic also gives an effect-size estimate that
predicts whether to expect a large or a modest-but-real ablation delta.

**This step is also the first check for the confound described in Section 4** — the same
"clinicians re-image volatile patients" pattern that creates data sparsity is mechanistically
related to diagnostic suspicion bias (see below). Keep both concerns in mind simultaneously
when reading the crosstab.

## 4. Required robustness check: diagnostic suspicion bias / "awareness deadlock"

This is a well-documented, real confound in sepsis prediction (Sarma et al., PLOS Digital
Health 2023, "Diagnostic suspicion bias and machine learning: breaking the awareness
deadlock for sepsis detection" — a review of 107 sepsis prediction algorithms found that
ZERO accounted for "informative observations," i.e. that the presence/timing of a
diagnostic test is not random but reflects clinician suspicion). CISepsis (Li et al. 2024,
Frontiers) is the closest attempt to debias this for notes specifically, using instrumental
variables + back-door adjustment, but even they still evaluate under a standard fixed
lookahead window rather than a strict pre-suspicion cutoff.

**Why this matters directly for SDCA:** when a clinician orders a fresh CXR on a suddenly-
volatile patient, that freshness may be informative partly because ordering behavior itself
encodes "a doctor just got worried" — not purely because of what the image physically shows.
SDCA could learn to exploit this confound (recent order = clinician suspicion = probably
septic) while appearing to be doing legitimate staleness-aware reasoning.

**Required check (add to Results, not optional):** evaluate under TWO protocols —
(1) the standard lead-time sweep already defined in Section 5, and
(2) a strict pre-suspicion-of-infection cutoff, using data available only before
`suspicion_time` (already computed in `label_sepsis3.py`) for each admission.
Report both. If SDCA's advantage shrinks substantially under the strict protocol, that
quantifies how much of the gain was suspicion-bias-driven. If it survives, that's strong
evidence the mechanism captures genuine physiological staleness reasoning — frame this
proactively in the paper as a check performed, not a weakness discovered by a reviewer.

## 5. Our solution: two modules (refined design — read carefully, this supersedes any
earlier "MLP gate multiplied into attention weights" description)

**Module 1 — SDCA (Staleness-Decay Cross-Attention), precision-weighted fusion form:**
NOT a raw learned scalar multiplied into attention weights after softmax (this breaks the
weights' sum-to-1 property, and renormalizing afterward silently distorts every other
modality's weight as an unintended side effect). Instead: model each modality's contribution
as an estimate with an associated uncertainty that GROWS with Δt and with volatility, fused
via inverse-variance weighting — the same principle a Kalman filter uses (trust the last
measurement less as time passes and as process noise increases). Implemented as an
**additive bias term in the attention logits, before softmax** (same family as relative
position encodings — a well-established, correctly-behaved pattern). This connects to
MGP-AttTCN's Gaussian-Process time-since-observation uncertainty already in our related
work — a nonstationary/heteroscedastic kernel whose length-scale depends on local volatility
gives the same joint signal in a theoretically grounded form, rather than an unexplained
black-box gate. FiLM-style representation modulation (`z' = γ(Δt, volatility) ⊙ z`, shrinking
the stale token's own representation before attention) is a valid, more expressive
alternative — keep as an ablation arm, not the primary form.

**Module 2 — SARL (Staleness-Aware Regularization Loss), directly-supervised form:**
NOT a vague regularizer hoping the decay function doesn't collapse. Instead: find patients
with two CXRs close together in time. Take the earlier one, artificially "age" it as if it
were the only observation available at the later timepoint, and supervise directly on the
ACTUAL delta in outcome probability between using the fresh vs. artificially-stale image.
Real gradient signal for "how much did staleness cost here," not emergent hope from the
primary task loss alone.

Both sit on top of **our own fusion backbone** — a standard multi-head cross-attention
design across the three modality token streams, built and owned by us. We do NOT fork
another paper's fusion architecture (MedPatch/FuseMoE/DrFuse) as our core model — we build
our own, prove it beats all baselines (each independently reproduced), then ablate
SDCA/SARL against OUR OWN architecture with them removed — never against a competitor's
architecture. See `docs/gap_and_solution.md` for the full explanation.

## 6. Closest related work (know these before implementing anything)

| Paper | What it does | What it's missing (our gap) |
|---|---|---|
| MedFuse | LSTM fusion, TS+CXR only | No notes, no staleness modeling |
| FuseMoE | mTAND + sparse MoE, TS+notes+CXR/ECG | Missingness handled as binary routing, no staleness |
| DrFuse | Disentangled shared/unique latent space, TS+CXR only | Doesn't scale past 2 modalities, no staleness |
| MedPatch | Token-level content confidence (calibration-based), TS+CXR+notes | Uses only most-recent CXR snapshot; confidence is static, not time-aware. Their own paper names multi-timepoint integration as future work — this is exactly our contribution. |
| DDL-CXR | Generates a new synthetic CXR via diffusion to fix staleness | CXR-only, expensive, can't generalize to notes |
| SepsisCalc / SepsisLab | Sepsis-specific, dynamic graph / uncertainty propagation | Structured EHR only, no notes/CXR; SepsisLab's uncertainty idea is single-modality |
| CISepsis | Causal debiasing of note/static-indicator confounders for sepsis | Text + static only, no CXR; still evaluates on standard lookahead, not strict pre-suspicion |
| MGP-AttTCN | GP-based time-since-observation uncertainty | Not multimodal; grounds our precision-weighted SDCA form |

## 7. Task definition (locked — do not change without team discussion)

**Cohort construction (filtering, applied BEFORE labeling) — see `preprocessing/label_sepsis3.py`:**
- One ICU stay per patient — first eligible stay only, not all admissions
- Adults only (age >= 18)
- Minimum ICU length of stay (>= 12h)
- Exclude admissions with insufficient chart/lab density (`MIN_VALID_OBS_HOURS`, currently
  configurable via `--min_valid_obs_hours`, default 5 raw hours per the task spec; SepsisCalc's
  own precedent, once corrected for their 3h binning, corresponds to ~15 raw hours — an
  explicit team decision, not silently baked in)
- Exclude admissions where sepsis criteria are already met at/before ICU admission
  (`excluded_reason = "sepsis_at_admission"`: suspicion_time <= intime AND hr=0 absolute
  SOFA >= 2) — no "early prediction" possible for these
- Split assignment is at the subject_id level, not hadm_id level — if a patient somehow
  contributes more than one admission, ALL of that patient's admissions go in the same split

**Labeling (Sepsis-3):**
- Suspicion of infection: earlier of (first antibiotic, first culture), valid only if the
  other event follows within 24h (antibiotics->cultures) or 72h (cultures->antibiotics)
- Sepsis onset: first time total SOFA increases by >=2 relative to baseline (SOFA at hr=0),
  within a -48h/+24h window around suspicion of infection
- Missing SOFA component -> assume healthy (0 points): operationalized as worst value in a
  trailing 24h window, coalesced to 0 only if nothing was observed in that window at all
- Reference implementations cross-checked (do not invent a new definition from scratch):
  MIT-LCP/mimic-code (`concepts/sepsis/`), alistairewj/sepsis3-mimic, yinchangchang/SepsisCalc
  (hours-since-admission alignment pattern; NOT their 3h binning — we use raw timestamps)

**Task formulation (how predictions are actually made):**
- NOT a fixed-window snapshot task (unlike MedFuse/FuseMoE/MedPatch's 48-IHM style). This is
  a continuous, rolling early-warning task, following SepsisCalc and SepsisLab's exact setup
  (both run hourly sliding predictions until diagnosis or discharge).
- A prediction timepoint is generated every hour per admission, starting after a short
  observation buffer, stopping at sepsis onset or discharge (no timepoints after onset).
- Single fixed prediction horizon W = 4 hours (matches SepsisCalc/SepsisLab directly): label
  = 1 at timepoint t if onset occurs within the next 4h, label = 0 otherwise. Negative
  patients are label = 0 at every valid hourly timepoint across their stay.
- ONE model is trained on this single task — NOT four separately trained lead-time models.
- The 2h/4h/6h/12h lead-time sweep (Table 2 / Figure 3) is produced by stratifying test
  predictions by TRUE distance-to-onset after training, not by training separate models.
- This hourly expansion happens dynamically in the dataset/dataloader code at training time
  (`experiments/`), NOT precomputed into `sepsis_labels.parquet` (stays admission-level).
- Negative-timepoint subsampling expected given severe class imbalance at timepoint level —
  keep all positive timepoints, downsample negatives to a documented ratio.

**Metrics:** AUROC, AUPRC (primary given class imbalance), ECE for calibration, reported
under BOTH the standard protocol and the strict pre-suspicion protocol (Section 4).

## 8. Repo map (what lives where)

```
sepsis-repo/
├── PROJECT_CONTEXT.md          <- this file
├── environment.yml              <- pinned conda env, same for everyone
├── docs/
│   ├── gap_and_solution.md      <- full gap explanation
│   ├── data_schema.md           <- locked Parquet schemas for every preprocessing output
│   ├── experiment_blueprint.md  <- full table/figure/ablation spec for the paper
│   └── verification_checklist.md <- how to verify AI-generated code (planned)
├── data/
│   ├── cohort/                  <- sepsis_labels.parquet, cohort_stats.json
│   ├── raw_links/                <- scripts pointing to physionet paths, NOT raw data
│   └── splits/                   <- fixed subject-level splits, versioned once
├── preprocessing/
│   ├── label_sepsis3.py          <- single source of truth, cohort + labeling
│   ├── ehr_extraction.py         <- Harutyunyan et al. 17-variable set, MIMIC-IV-ported
│   ├── notes_extraction.py       <- full timestamped RR+DN sequence, MedPatch-style chunking
│   ├── cxr_linking.py            <- full timestamped CXR sequence, not just most-recent
│   └── normalizers.py            <- shared normalization, used by every model
├── models/
│   ├── encoders/                 <- TS/notes/CXR encoders, standard/off-the-shelf on purpose
│   ├── baselines/                 <- MedFuse, FuseMoE, MedPatch, DrFuse reproductions
│   └── ours/
│       ├── backbone.py            <- our own fusion architecture
│       ├── sdca.py                <- precision-weighted fusion, additive attention-logit bias
│       └── sarl.py                <- near-duplicate-CXR supervised staleness cost
├── experiments/
│   ├── configs/                   <- one YAML per run: task, lead-time, modality combo, seed
│   ├── diagnostics/                <- Step 0 signal-verification scripts (Section 3)
│   ├── train.py
│   ├── evaluate.py                 <- AUROC/AUPRC/ECE, both eval protocols, bootstrapped CIs
│   └── stress_tests/               <- missingness sweep, staleness-injection sweep
├── notebooks/                      <- exploratory only, nothing load-bearing lives here
└── results/                        <- versioned CSVs/JSON of metrics per run, no model weights
```

## 9. Current status
> Update this section every week so everyone (and every AI assistant) knows what's done.

- [x] Cohort construction / Sepsis-3 labeling (`label_sepsis3.py` — bugs found + fixed via
      team review: dead-code sepsis_at_admission check, GCS convention, vectorized SOFA
      scoring, min_valid_obs_hours ambiguity resolved as an explicit flag)
- [ ] EHR/notes/CXR preprocessing pipelines
- [ ] Step 0 signal-verification diagnostic (Section 3) — MUST pass before Module 1/2 build
- [ ] Baseline reproductions (MedFuse / FuseMoE / MedPatch / DrFuse)
- [ ] Our backbone architecture
- [ ] SDCA implementation (precision-weighted form)
- [ ] SARL implementation (near-duplicate-CXR supervised form)
- [ ] Main results under standard protocol (Table 2 / Figure 3)
- [ ] Main results under strict pre-suspicion protocol (Section 4)
- [ ] Ablations (Table 3)
- [ ] Stress tests (Table 4 / Figure 4)

## 10. Rules that must never be broken
1. Every baseline number reported must come from **our own re-run** on our cohort — never
   copy numbers from the original papers.
2. `preprocessing/label_sepsis3.py` is the single source of truth for sepsis labels — if you
   need a different definition, discuss with the team first, don't fork the logic silently.
3. Ablations (Table 3) are performed against **our own architecture**, not a baseline's.
4. Don't merge to `main` without a reviewed PR — see `CONTRIBUTING.md`.
5. **Every model — every baseline reproduction AND our own architecture — consumes data
   from the SAME `preprocessing/` pipeline.** Never let a baseline bring its own separate
   extraction/normalization code. Reference a baseline's variable set/normalization/split
   conventions when building our shared pipeline — just build it once, here, for everyone.
6. **Do not build SDCA/SARL until the Step 0 diagnostic (Section 3) passes.** A gating
   mechanism can't learn a signal that isn't demonstrably in the data.
7. **Every reported result includes the strict pre-suspicion evaluation (Section 4)
   alongside the standard protocol.** Never report only the standard-protocol number.
8. For any AI-generated file handling non-trivial logic (cohort/labels, SOFA, fusion math):
   require an explicit ASSUMPTIONS section, verify cited sources against the actual source
   file at least once, run on a small sample first, and get a second independent AI/human
   review before trusting it — see the verification process established for
   `label_sepsis3.py` as the template.
