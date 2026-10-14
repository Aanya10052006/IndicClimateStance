
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "raw_comments.csv"
SAMPLE_FILES = [
    ROOT / "data" / "processed" / "annotation_sample.csv",
    ROOT / "data" / "processed" / "annotation_sample_2.csv",
]
CANDIDATES_FILE = ROOT / "data" / "processed" / "against_candidates.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "remaining_review.csv"

raw = pd.read_csv(RAW_FILE, encoding="utf-8-sig", keep_default_na=False)
candidates = pd.read_csv(
    CANDIDATES_FILE, encoding="utf-8-sig", keep_default_na=False
)

excluded_ids = set()

for file in SAMPLE_FILES:
    sample = pd.read_csv(file, encoding="utf-8-sig", keep_default_na=False)
    excluded_ids.update(sample["comment_id"].astype(str))

excluded_ids.update(candidates["comment_id"].astype(str))

raw["comment_id"] = raw["comment_id"].astype(str)
remaining = raw[~raw["comment_id"].isin(excluded_ids)].copy()

remaining["stance"] = ""
remaining["annotation_notes"] = ""

columns = [
    "comment_id",
    "video_id",
    "language",
    "comment_text",
    "stance",
    "annotation_notes",
]
columns = [col for col in columns if col in remaining.columns]
remaining = remaining[columns]

remaining.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Remaining review file created.")
print(f"Raw comments: {len(raw)}")
print(f"Excluded comments: {len(excluded_ids)}")
print(f"Comments to review: {len(remaining)}")
print(f"Saved to: {OUTPUT_FILE}")