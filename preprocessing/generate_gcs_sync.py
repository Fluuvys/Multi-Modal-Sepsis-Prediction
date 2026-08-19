import pandas as pd

meta = pd.read_csv("/home/fluuvys-main/Research/Multi modal sepsis prediction/Data/cxr/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz")  # adjust path if needed
cohort = pd.read_parquet("../data/cohort/sepsis_labels.parquet")
cohort = cohort[cohort["excluded_reason"].isna()]

# only subjects that actually appear in the CXR metadata -- this is the
# real fix, avoids ~44,700 expected-but-noisy failures for subjects with
# zero CXR studies
subjects_with_cxr = sorted(
    int(s) for s in set(meta["subject_id"]) & set(cohort["subject_id"])
)

base = "/home/fluuvys-main/sepsis_proj/Data/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
project_id = "project-3a8c4ca8-da13-47aa-945"

with open("gcs_sync_commands.sh", "w") as f:
    for s in subjects_with_cxr:
        shard = str(s)[:2]
        f.write(
            f"gsutil -u {project_id} -m rsync -r "
            f"'gs://mimic-cxr-jpg-2.1.0.physionet.org/files/p{shard}/p{s}' "
            f"'{base}/p{shard}/p{s}'\n"
        )
print(f"{len(subjects_with_cxr)} subjects with CXR data -> "
      f"{len(subjects_with_cxr)} sync commands written")
