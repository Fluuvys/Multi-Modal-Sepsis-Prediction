# The Gap and Our Solution (detailed version)

## The gap

Every existing clinical multimodal fusion architecture decides how much to trust a
modality's observation based on:
1. **Whether it exists** — present vs. missing (FuseMoE's routing, DrFuse's shared-latent
   fallback, MedFuse's imputation).
2. **What it contains** — content-based confidence (MedPatch's calibrated token confidence).

**None of them decide based on how old the observation is relative to how fast the patient is
currently changing.** A CXR taken 30 hours ago is treated identically whether the patient has
been stable that whole time or has been deteriorating for the last 4 hours — because "present
+ confident" is the same input either way, regardless of how much the patient's state has
moved since that data was captured.

The one architecture that does engage with staleness, DDL-CXR, fixes it by generating a
brand-new synthetic CXR via latent diffusion conditioned on the EHR trajectory — expensive,
and structurally CXR-only (you cannot diffusion-generate a new clinical note the same way).

So: missingness is solved (binary), content-confidence is solved, and staleness is solved
only for images, only through expensive generation. There is no general, lightweight,
modality-agnostic mechanism where fusion weight is parameterized jointly by **elapsed time**
and **patient volatility**.

## Why this specific gap (not encoders, not missing-data handling)

- Irregular time-series encoding is solved (mTAND, ContiFormer, bi-axial transformers).
- Missing-modality handling is solved at the binary level (FuseMoE, DrFuse).
- Modality-specific encoders (BioBERT, ViT, DenseNet) are a solved, boring problem.
- The unsolved intersection: translating (time since observation, current patient trajectory)
  into a fusion weight. Nobody combines these two signals into one learned mechanism.

## The solution: two named components

### SDCA — Staleness-Decay Cross-Attention
A learned decay term multiplies into fusion/attention weights before tokens are combined.
Input: (Δt since capture, a rolling volatility statistic from the TS stream). A 12-hour-old
note on a stable patient keeps high trust; the same 12-hour-old note on a deteriorating
patient gets automatically down-weighted. Modality-agnostic — works identically for CXR,
notes, or labs, unlike DDL-CXR.

### SARL — Staleness-Aware Regularization Loss
An auxiliary training term preventing the decay function from collapsing to something
degenerate under pure task-loss optimization (e.g. ignoring elapsed time entirely, or staying
overconfident when only stale data is available). Penalizes overconfidence specifically when
available modalities are stale.

Both sit on top of **our own fusion backbone** (standard multi-head cross-attention across
the three modality token streams) — not forked from MedPatch/FuseMoE/DrFuse's specific
architecture. We prove our full model (backbone + SDCA + SARL) beats all baselines, then
ablate SDCA/SARL against our own architecture, same as every paper in our related work does
with their own novel components.

## Why it's solvable and provable, not just aspirational

- Small, surgical addition to an existing cross-attention layer — no new backbone required.
- Cleanly ablatable: on/off, learned/fixed decay, per-modality/shared, time-only/volatility-
  aware — each isolates one design decision.
- Directly comparable against DDL-CXR (same symptom, different fix) at a fraction of the
  compute, and against MedPatch (whose own paper names multi-timepoint integration as future
  work — this is literally that future work, done).

## If it works, what it buys us

1. Better early warnings specifically when a patient is deteriorating and their data is going
   stale — the exact failure mode current models can't see.
2. Predictions that degrade gracefully as data ages or modalities go missing, instead of
   failing suddenly.
3. Honest uncertainty for free — wider confidence intervals when only stale data is available.
4. Generalizes across modalities, unlike DDL-CXR's CXR-only fix.
5. A clean ablation story for reviewers: turn off SDCA, see one failure mode; turn off SARL,
   see a different one.
