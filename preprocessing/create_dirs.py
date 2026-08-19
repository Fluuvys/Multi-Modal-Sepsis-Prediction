import os

with open("gcs_sync_commands.sh") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # line looks like: gsutil ... -r 'SOURCE' 'DEST'
        # splitting on ' gives: [..., SOURCE, ' ', DEST]  -> DEST is second-to-last
        parts = line.split("'")
        dest = parts[-2]
        os.makedirs(dest, exist_ok=True)

print("done creating destination directories")