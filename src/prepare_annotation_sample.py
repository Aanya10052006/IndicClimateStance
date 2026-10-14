
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "raw_comments.csv"
OUTPUT = ROOT / "data" / "processed" / "annotation_sample.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig", keep_default_na=False)

# Select a reproducible random sample for initial manual review.
sample = df.sample(n=min(100, len(df)), random_state=42).copy()

sample["stance"] = ""
sample["annotation_notes"] = ""

sample.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Created annotation sample with {len(sample)} comments.")
print(f"Saved to: {OUTPUT}")
print("\nLabel each comment using:")
print("Favour = supports the claim")
print("Against = rejects the claim")
print("None = neutral, unrelated, or unclear stance")