"""
train.py -- main training entrypoint

Reads a config from experiments/configs/, trains any model (baseline or ours) on the
sepsis-onset cohort at a specified lead time, saves results to results/.

TODO:
    [ ] Argparse: --config path, --seed
    [ ] Load cohort + labels + chosen modality combo
    [ ] Instantiate model from config (baseline or models/ours/backbone.py + sdca/sarl)
    [ ] Standard train loop with early stopping
    [ ] Save metrics to results/<run_name>.json (never overwrite -- append run metadata)
"""

# TODO: implementation goes here
