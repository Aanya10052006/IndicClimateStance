from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final"

# Read existing training data
df = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
    keep_default_na=False
)

# Validate labels
valid_labels = {"Favour", "Against", "None"}

df["stance"] = df["stance"].astype(str).str.strip()

invalid = df[~df["stance"].isin(valid_labels)]

if not invalid.empty:
    raise ValueError(
        f"Found {len(invalid)} invalid stance labels."
    )

# Split 80% training and 20% validation
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["stance"]
)

# Save original training data as a backup
backup_path = DATA_DIR / "train_before_validation.csv"

if backup_path.exists():
    raise FileExistsError(
        "Backup already exists. Inspect it before rerunning."
    )

df.to_csv(
    backup_path,
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    DATA_DIR / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    DATA_DIR / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Dataset split completed!")

print("\nTraining shape:", train_df.shape)
print("Validation shape:", val_df.shape)

print("\nTraining stance distribution:")
print(train_df["stance"].value_counts())

print("\nValidation stance distribution:")
print(val_df["stance"].value_counts())
