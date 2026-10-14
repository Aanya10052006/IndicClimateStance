
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "final" / "labelled_dataset.csv"

FILES = [
    PROCESSED / "annotation_sample.csv",
    PROCESSED / "annotation_sample_2.csv",
    PROCESSED / "against_candidates.csv",
    PROCESSED / "remaining_review.csv",
]

VALID_LABELS = {"Favour", "Against", "None"}
all_frames = []

for file in FILES:
    if not file.exists():
        raise SystemExit(
            f"\nMissing file: {file}\n"
            "Check that you placed all four CSV files in data/processed."
        )

    df = pd.read_csv(file, encoding="utf-8-sig", keep_default_na=False)

    # The generated review CSV uses suggested_stance.
    if "suggested_stance" in df.columns:
        if "stance" in df.columns:
            # Keep the final reviewed label in stance if present.
            df["stance"] = df["stance"].where(
                df["stance"].astype(str).str.strip() != "",
                df["suggested_stance"]
            )
        else:
            df["stance"] = df["suggested_stance"]

    required = {"comment_id", "video_id", "comment_text", "stance"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"\n{file.name} is missing columns: {sorted(missing)}"
        )

    df["stance"] = df["stance"].astype(str).str.strip()
    df["comment_id"] = df["comment_id"].astype(str).str.strip()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    all_frames.append(df)

combined = pd.concat(all_frames, ignore_index=True, sort=False)

# Check for duplicate IDs before removing anything.
duplicate_rows = combined[combined.duplicated("comment_id", keep=False)]

print("\nRows before duplicate check:", len(combined))
print("Unique comment IDs:", combined["comment_id"].nunique())
print("Rows involved in duplicate IDs:", len(duplicate_rows))

if not duplicate_rows.empty:
    print("\nDuplicate IDs found. Resolve them before training:")
    print(
        duplicate_rows[["comment_id", "video_id", "stance"]]
        .to_string(index=False)
    )
    raise SystemExit("Merge stopped because duplicate IDs were found.")

missing_labels = combined[
    ~combined["stance"].isin(VALID_LABELS)
]

if not missing_labels.empty:
    print("\nMissing or invalid labels:")
    print(
        missing_labels[["comment_id", "comment_text", "stance"]]
        .to_string(index=False)
    )
    raise SystemExit("Merge stopped. Correct these labels first.")

if (combined["comment_text"] == "").any():
    raise SystemExit("Empty comments found. Inspect them before training.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nDataset merged successfully!")
print("Saved to:", OUTPUT)
print("\nLabel distribution:")
print(combined["stance"].value_counts().to_string())

print("\nLabel distribution by source-video language:")
if "language" in combined.columns:
    print(pd.crosstab(combined["language"], combined["stance"]))

print("\nTotal labelled comments:", len(combined))