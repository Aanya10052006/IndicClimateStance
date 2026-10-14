
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "data" / "final"

DATASET = FINAL / "labelled_dataset.csv"
REVIEW = PROCESSED / "against_review.csv"
OUTPUT = FINAL / "labelled_dataset_reviewed.csv"

df = pd.read_csv(DATASET, encoding="utf-8-sig", keep_default_na=False)
review = pd.read_csv(REVIEW, encoding="utf-8-sig", keep_default_na=False)

required = {"comment_id", "reviewed_stance"}
if not required.issubset(review.columns):
    raise SystemExit("Review file is missing required columns.")

review = review[review["reviewed_stance"].astype(str).str.strip() != ""].copy()
review["reviewed_stance"] = review["reviewed_stance"].astype(str).str.strip()

valid = {"Favour", "Against", "None"}
if not set(review["reviewed_stance"]).issubset(valid):
    raise SystemExit("Invalid reviewed stance found.")

if review["comment_id"].duplicated().any():
    raise SystemExit("Duplicate comment IDs in review file.")

label_map = review.set_index("comment_id")["reviewed_stance"]
mask = df["comment_id"].isin(label_map.index)

print("Comments matched for review:", int(mask.sum()))

df.loc[mask, "stance"] = df.loc[mask, "comment_id"].map(label_map)

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nSaved reviewed dataset to:", OUTPUT)
print("\nUpdated label distribution:")
print(df["stance"].value_counts().to_string())
