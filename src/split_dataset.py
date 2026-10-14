
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "final" / "labelled_dataset_reviewed.csv"
OUTPUT_DIR = ROOT / "data" / "final"

df = pd.read_csv(INPUT, encoding="utf-8-sig", keep_default_na=False)

# Check required columns.
required = {"comment_id", "video_id", "comment_text", "stance"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

if df["video_id"].astype(str).str.strip().eq("").any():
    raise SystemExit("Some comments have no video_id.")

# First, reserve approximately 20% of videos for testing.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["video_id"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# Ensure no source video occurs in both splits.
train_videos = set(train_df["video_id"])
test_videos = set(test_df["video_id"])

assert train_videos.isdisjoint(test_videos)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(
    OUTPUT_DIR / "train.csv", index=False, encoding="utf-8-sig"
)
test_df.to_csv(
    OUTPUT_DIR / "test.csv", index=False, encoding="utf-8-sig"
)

print("Video-wise split complete.")
print(f"Training comments: {len(train_df)}")
print(f"Testing comments: {len(test_df)}")
print(f"Training videos: {len(train_videos)}")
print(f"Testing videos: {len(test_videos)}")

print("\nTraining label distribution:")
print(train_df["stance"].value_counts().to_string())

print("\nTesting label distribution:")
print(test_df["stance"].value_counts().to_string())

print("\nVideos shared between train and test:",
      len(train_videos.intersection(test_videos)))