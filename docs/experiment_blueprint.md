# Staleness-Decay Fusion for Early Sepsis Prediction — Paper Blueprint

Core gap: existing clinical multimodal fusion architectures decide how much to trust a
modality's observation based on whether it exists (present/absent) or what it contains
(content confidence) — none decide based on how old it is relative to how fast the patient
is currently changing. Two named components fix this: **SDCA** (Staleness-Decay
Cross-Attention) and **SARL** (Staleness-Aware Regularization Loss), on top of an
otherwise standard fusion backbone.

Sections marked **[ADDED]** below weren't in your original outline — I filled them in
because a Q1 submission needs them, and flagged exactly why each one is there.

---

## Abstract
One paragraph. Structure: (1) clinical problem — early sepsis prediction needs multiple
modalities, but they arrive asynchronously; (2) the gap — no fusion mechanism weights
observations by elapsed-time + patient volatility, only by presence or content-confidence;
(3) your fix — SDCA + SARL; (4) headline result — beats baselines, gap widens at longer
lead times, at a fraction of DDL-CXR's compute.

---

## 1. Introduction

**1.1 Motivate the staleness gap with a concrete clinical example**
Open with the two-patient example: identical 30-hour-old "clean" CXR, one patient stable,
one deteriorating for the last 4 hours. Every current fusion model treats these identically.
This is your hook — keep it to one vivid paragraph, no citations yet.

**1.2 Outline the solution/method**
One paragraph: SDCA makes fusion weight a learned function of (elapsed time since capture,
current trajectory volatility); SARL keeps that function from collapsing to something
degenerate under pure task-loss pressure. Name them here for the first time.

**1.3 Contributions (bulleted, exactly 3)**
1. A general, modality-agnostic staleness-aware fusion mechanism (SDCA), unlike DDL-CXR's
   CXR-only generative fix.
2. A matched regularization loss (SARL) that keeps the decay function clinically sensible.
3. First tri-modal (irregular EHR + notes + CXR) evaluation on **sepsis onset** specifically
   (existing tri-modal fusion work targets mortality/phenotyping, not sepsis).

---

## 2. Related Work

**2.1 Sepsis prediction papers (ML/DL)**
SepsisCalc (dynamic temporal graph), SepsisLab (uncertainty propagation, active sensing),
MIMIC-Sepsis benchmark, PhysioNet-2019-style transformer/LSTM papers. All structured-EHR
only — no fusion problem exists here because there's only one modality.

**2.2 Multimodal models for sepsis specifically**
SepsiGraph (EHR + CXR, graph-based, sepsis onset) — closest sepsis-specific multimodal
work, but no notes modality, and "temporal proximity" edges are not the same as a learned
decay function. This is a thin section — say so; it's part of your novelty argument (tri-modal
sepsis work barely exists).

**2.3 Multimodal fusion, general EHR (not sepsis-specific)**
This is where the real architectural competitors live — organize by what each one gets right
and what it misses, not chronologically:
- MedFuse — LSTM sequence fusion, no notes, no staleness modeling.
- FuseMoE — mTAND + sparse MoE, strong irregularity handling, fusion still purely
  downstream-loss-driven, missingness handled as binary via routing.
- DrFuse — disentangled shared/unique latent space; doesn't scale past 2 modalities.
- MedPatch — token-level *content* confidence via calibration, not time; explicitly limited
  to single most-recent CXR per patient, names multi-timepoint integration as their own
  future work. **This is your strongest, most direct citation** — quote their limitation
  section closely.
- DDL-CXR — solves staleness via generative latent diffusion; CXR-only, expensive, can't
  generalize to notes.

End this section with one paragraph explicitly stating the gap (your "clear as ICCR-Net"
paragraph from earlier), positioned as the synthesis of 2.1–2.3.

---

## 3. Proposed Method

**3.1 Notations**
Define once, use consistently: patient index, modality set {ehr, notes, cxr}, token/observation
timestamp, prediction time, elapsed time Δt = (prediction time − observation time), trajectory
volatility statistic (define precisely — e.g., a rolling variance or rate-of-change over the TS
stream in a fixed lookback window), fusion weight, decay function parameters.

**3.2 Overview of method**
One diagram (Fig. 2 below), one paragraph walking through the pipeline: frozen/standard
modality encoders → tokens with timestamps preserved → SDCA computes a decay-adjusted
fusion weight per token → cross-attention/fusion → decoder → prediction. State plainly that
the encoders are intentionally standard/off-the-shelf (novelty lives in SDCA and SARL), but
the fusion backbone itself is your own assembled architecture (standard multi-head
cross-attention + gating, built and owned by you) — not a fork of MedPatch, FuseMoE, or
DrFuse's specific pipeline. Baselines are independently reproduced for comparison in
Table 2; ablations in Table 3 are performed against your own architecture only, matching
standard practice (MedFuse/FuseMoE/MedPatch/DrFuse each ablate their own novel
components against their own full model, never against a competitor's).

**3.3 Each new component**
- **SDCA**: exact functional form of the decay gate, how Δt and volatility combine, how it's
  learned jointly with the rest of the network, per-modality vs. shared parameterization.
- **SARL**: the auxiliary loss term, what failure mode it prevents (e.g., overconfident
  predictions when only stale data is available), how it's weighted against the main task loss.

---

## 4. Experiments

**4.1 Experimental settings (datasets and preprocessing)**
MIMIC-IV (EHR, 17-variable standard extraction), MIMIC-IV-Note (radiology reports +
discharge notes, full timestamped sequence — NOT just most-recent, unlike MedPatch),
MIMIC-CXR (full sequence per patient, joined by subject/admission ID). Sepsis-3 cohort
construction (suspicion of infection + SOFA delta), lead-time task definition (2h/4h/6h/12h
before onset), train/val/test split, cohort statistics table (Table 1), modality-availability-
over-time motivation figure (Fig. 1).

**4.2 Baselines** **[ADDED — you'll need this before Results makes sense]**
Unimodal (TS/notes/CXR), concatenation+carry-forward, MedFuse, FuseMoE, MedPatch,
DDL-CXR-lite (adapted to sepsis). All re-run on your cohort/labels, never cited from their
original papers.

**4.3 Evaluation metrics and implementation details** **[ADDED — reviewers expect this
as its own subsection, not buried in a table caption]**
AUROC, AUPRC (primary given class imbalance), ECE/MCE for calibration. Training details:
optimizer, learning rate search, early stopping, number of seeds, how confidence intervals
were computed (bootstrapping, matching MedPatch's approach).

**4.4 Main results** **[ADDED]**
Table 2 (headline comparison table across models × lead times × modality combos) and
Figure 3 (the lead-time sweep line plot — your strongest visual argument).

**4.5 Ablation studies** **[ADDED]**
Table 3: full model vs. w/o SDCA, w/o SARL, fixed vs. learned decay, per-modality vs. shared
decay, time-only vs. volatility-aware decay, and **your own base fusion backbone with both
removed** (a standard multi-head cross-attention design you assemble and own — NOT
MedPatch's or FuseMoE's architecture). This last row is your internal floor, distinct from the
external baselines in Table 2. Ablations isolate what SDCA/SARL add on top of your own
architecture, exactly as MedPatch/FuseMoE/DrFuse each ablate their own novel components
against their own full model — never against a competitor's architecture.

**4.6 Robustness: missingness and staleness stress tests** **[ADDED — this is the direct
test of your actual claim, don't let it get cut for space]**
Table 4 (missingness sweep at 0/20/50/80%; staleness injection at 6h/24h/48h/72h) and
Figure 4 (degradation curves vs. baselines).

**4.7 Efficiency comparison** **[ADDED]**
Table 5: parameters, training/inference wall-clock, vs. DDL-CXR-lite specifically — your
concrete "same problem, fraction of the compute" claim.

**4.8 Calibration** **[ADDED]**
Table 6: ECE/MCE/Brier score across models.

**4.9 Interpretability** **[ADDED]**
Table 7 (learned decay half-life per modality — a cheap, interesting standalone finding) and
Figure 5 (2–3 case-study patient trajectories showing decay weights dropping before
deterioration becomes visible in vitals). State explicitly: don't over-claim causal
interpretation from attention/decay weights.

---

## 5. Discussion / Limitations **[ADDED — near-mandatory for Q1 clinical-ML now]**

- Sensitivity of results to the exact Sepsis-3 operationalization (cite that this affects
  results more than model choice — document your definition exhaustively).
- Single-center (MIMIC-IV/BIDMC) — state what could/couldn't be externally validated.
- Subgroup breakdown (Table 8: age/sex/ethnicity) — report honestly, including any gaps.
- Name joint cross-modal pretraining (the deepest gap from your earlier analysis) as the
  natural next paper — this preempts a reviewer asking "why didn't you pretrain jointly."

---

## 6. Conclusion
Recap the gap in one sentence, the fix in one sentence, the headline number, and one
forward-looking sentence pointing at the pretraining direction named in Discussion.

---

## Appendix (optional, for overflow)
- Table 9: external validation (eICU/AmsterdamUMCdb), if feasible — if not, this becomes a
  named limitation instead, not a silently dropped section.
- Full lead-time sweep numbers for every ablation (main text shows only one representative
  lead time, e.g. 6h, to stay readable).

---

## Execution order (unchanged from before, mapped to this structure)
1. Cohort lock → §4.1, Table 1, Fig. 1.
2. Unimodal baselines → sanity check only.
3. Reproduce MedFuse/FuseMoE/MedPatch on your cohort → §4.2 skeleton.
4. Drop SDCA into best baseline's fusion layer, single lead time → go/no-go checkpoint.
5. Full lead-time sweep → §4.4 complete.
6. Ablations → §4.5.
7. Stress tests → §4.6.
8. Efficiency comparison → §4.7.
9. Calibration → §4.8.
10. Decay half-life + case studies → §4.9.
11. Subgroups + external validation → §5, Appendix.
12. Freeze results, write.

**Non-negotiables:** every baseline number is your own re-run; Sepsis-3 definition documented
exhaustively; Table 4 stress tests must exist; state explicitly what you could NOT validate.
