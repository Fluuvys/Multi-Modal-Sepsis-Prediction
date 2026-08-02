# Baselines

Each baseline is reproduced independently, on OUR sepsis-onset cohort/labels -- never cite
numbers from the original papers directly (see PROJECT_CONTEXT.md rule #1).

- medfuse.py   -- TS + CXR only (2 modalities), report explicitly labeled as partial
- fusemoe.py   -- TS + notes + CXR, mTAND + sparse MoE
- medpatch.py  -- TS + notes + CXR, token-level confidence + missingness module
- drfuse.py    -- TS + CXR only (2 modalities), disentangled shared/unique representation

TODO: assign one baseline per person as a separate GitHub issue. Each should be its own PR,
reviewed independently, before moving to models/ours/.
