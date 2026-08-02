# Staleness-Decay Fusion for Early Sepsis Prediction

Multimodal (irregular EHR time-series + clinical notes + chest X-ray) deep learning for
**sepsis onset prediction** on MIMIC-IV / MIMIC-CXR / MIMIC-IV-Note.

**Start here:** [`PROJECT_CONTEXT.md`](./PROJECT_CONTEXT.md) — the gap, the solution, the
task definition, and the full repo map, all in one file. Read it before opening any code, and
paste it into your AI assistant before asking for coding help.

**Contributing:** see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the branching/PR/review
workflow and how tasks are assigned.

**Full experiment plan:** [`docs/experiment_blueprint.md`](./docs/experiment_blueprint.md) —
every table and figure the paper needs, mapped to the exact experiment that produces it.

## Quick status
See `PROJECT_CONTEXT.md` §7 for the live milestone checklist.

## Setup
```
# TODO: fill in once environment/dependencies are finalized
pip install -r requirements.txt
```

## Data access
This repo does not contain any patient data. You need your own credentialed PhysioNet access
to MIMIC-IV, MIMIC-CXR, and MIMIC-IV-Note. See `data/raw_links/` for scripts pointing to
expected local paths — never commit raw data or patient-level identifiers to this repo.
