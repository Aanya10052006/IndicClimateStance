
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "raw_comments.csv"
OLD_SAMPLE = ROOT / "data" / "processed" / "annotation_sample.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "annotation_sample_2.csv"

# Load the collected comments and the first annotation sample
raw = pd.read_csv(RAW_FILE, encoding="utf-8-sig", keep_default_na=False)
old = pd.read_csv(OLD_SAMPLE, encoding="utf-8-sig", keep_default_na=False)

# Exclude comments already present in the first sample
used_ids = set(old["comment_id"].astype(str))
remaining = raw[~raw["comment_id"].astype(str).isin(used_ids)].copy()

# Select up to 150 additional comments reproducibly
sample_size = min(150, len(remaining))
sample = remaining.sample(n=sample_size, random_state=43).copy()

# Add empty annotation columns
sample["stance"] = ""
sample["annotation_notes"] = ""

sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Additional annotation sample created!")
print(f"Comments in original dataset: {len(raw)}")
print(f"Comments already sampled: {len(used_ids)}")
print(f"Comments available for annotation: {len(remaining)}")
print(f"New comments selected: {len(sample)}")
print(f"Saved to: {OUTPUT_FILE}")
