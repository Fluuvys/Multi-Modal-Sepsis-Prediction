# Data Schema

All shared preprocessing outputs are **Parquet**, not pickle -- columnar, queryable,
cross-tool, versions safely. Images stay as files on disk, referenced by path only.

This is a **lossless master format** -- full timestamped sequences, nothing binned or
truncated. Individual models (baselines or ours) each apply their OWN reduction/adapter
on top of this master data if they need something more reduced (e.g. MedPatch's
hourly-binning + most-recent-only behavior). Never bake a model-specific reduction into
shared preprocessing -- see PROJECT_CONTEXT.md rule #5.

## `data/cohort/sepsis_labels.parquet`
One row per admission.
| column | type | notes |
|---|---|---|
| subject_id | int | |
| hadm_id | int | join key used by every other file |
| sepsis_onset_time | datetime, nullable | null if excluded/negative |
| sofa_at_onset | float, nullable | |
| label | int (0/1) | the actual training target |
| excluded_reason | string, nullable | null if included |
| split | string | "train" / "val" / "test", fixed once |

## `data/cohort/ehr_timeseries.parquet`
Long format, one row per observation (NOT one file per patient).
| column | type | notes |
|---|---|---|
| hadm_id | int | join key |
| timestamp | datetime | raw, unbinned |
| variable_name | string | one of the standard 17-variable set |
| value | float | |
| hours_before_onset | float | precomputed from sepsis_labels.sepsis_onset_time; SDCA's Δt term reads this directly |

## `data/cohort/notes.parquet`
| column | type | notes |
|---|---|---|
| hadm_id | int | |
| note_id | string | |
| note_type | string | "RR" (radiology report) or "DN" (discharge note) |
| timestamp | datetime | |
| raw_text | string | NOT pre-embedded -- embedding happens per-model, downstream |
| hours_before_onset | float | |

## `data/cohort/cxr_metadata.parquet`
| column | type | notes |
|---|---|---|
| hadm_id | int | |
| study_id | string | |
| dicom_id | string | |
| timestamp | datetime | |
| image_path | string | points to actual JPG on disk, not embedded here |
| hours_before_onset | float | |

## Per-model adapters (where format translation happens)

Shared master data → each model's own thin adapter → model-specific input shape.

- `models/baselines/medpatch.py`: bins EHR hourly, takes only the row with minimum
  `hours_before_onset` per modality per patient -- reproduces their actual behavior.
- `models/baselines/fusemoe.py`: feeds raw timestamps into mTAND (no binning needed).
- `models/ours/backbone.py` + `sdca.py`: consumes the master format as-is, full
  sequence, real timestamps -- no reduction. This is the point of the contribution.

Writing this adapter is part of each baseline/model's own file -- do not add
model-specific logic to anything under `preprocessing/`.
