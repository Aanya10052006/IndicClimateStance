
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "final" / "labelled_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "against_review.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig", keep_default_na=False)

against = df[df["stance"].astype(str).str.strip() == "Against"].copy()

against["reviewed_stance"] = ""
against["review_notes"] = ""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
against.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Against comments to review: {len(against)}")
print(f"Saved to: {OUTPUT}")
print("\nReview each comment using the original context where possible.")