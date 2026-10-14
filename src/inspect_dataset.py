
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "raw_comments.csv"

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", keep_default_na=False)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"\nTotal rows: {len(df)}")
print(f"Unique comment IDs: {df['comment_id'].nunique()}")
print(f"Duplicate comment IDs: {df['comment_id'].duplicated().sum()}")
print(f"Empty comments: {(df['comment_text'].str.strip() == '').sum()}")

print("\nComments by language:")
print(df["language"].value_counts())

print("\nComments per video:")
print(
    df.groupby(["language", "video_id"])
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nComment length statistics:")
lengths = df["comment_text"].str.len()
print(lengths.describe().round(2).to_string())

print("\n10 sample comments:")
for _, row in df.sample(
    n=min(10, len(df)),
    random_state=42
).iterrows():
    print(f"\nLanguage: {row['language']}")
    print(f"Comment: {row['comment_text']}")

print("\nInspection complete.")