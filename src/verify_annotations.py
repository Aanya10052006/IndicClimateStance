
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "processed" / "annotation_sample.csv",
    ROOT / "data" / "processed" / "annotation_sample_2.csv",
]

valid_labels = {"Favour", "Against", "None"}
all_samples = []

for file in FILES:
    if not file.exists():
        print(f"\nERROR: File not found: {file}")
        raise SystemExit(1)

    df = pd.read_csv(file, encoding="utf-8-sig", keep_default_na=False)
    df["stance"] = df["stance"].astype(str).str.strip()

    missing = (df["stance"] == "").sum()
    invalid = df.loc[
        (df["stance"] != "") & ~df["stance"].isin(valid_labels)
    ]

    print(f"\nFile: {file.name}")
    print(f"Total comments: {len(df)}")
    print(f"Missing labels: {missing}")
    print(f"Invalid labels: {len(invalid)}")

    if not invalid.empty:
        print(invalid[["comment_text", "stance"]].to_string(index=False))

    all_samples.append(df)

combined = pd.concat(all_samples, ignore_index=True)

print("\n" + "=" * 40)
print("COMBINED ANNOTATION SUMMARY")
print("=" * 40)
print(f"Total annotated rows: {len(combined)}")
print(f"Unique comment IDs: {combined['comment_id'].nunique()}")

print("\nLabel distribution:")
print(combined["stance"].value_counts(dropna=False).to_string())

print("\nVerification complete.")