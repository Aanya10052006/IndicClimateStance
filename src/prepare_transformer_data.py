from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

LABEL2ID = {
    "Against": 0,
    "Favour": 1,
    "None": 2
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def prepare_data(filename):
    df = pd.read_csv(
        DATA_DIR / filename,
        encoding="utf-8-sig",
        keep_default_na=False
    )

    required = {"comment_text", "stance", "language"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df["stance"] = df["stance"].astype(str).str.strip()

    df = df[
        df["comment_text"].ne("")
        & df["stance"].isin(LABEL2ID)
    ].copy()

    df["label"] = df["stance"].map(LABEL2ID)

    tokens = tokenizer(
        df["comment_text"].tolist(),
        truncation=True,
        padding=False,
        max_length=128
    )

    print(f"\n{filename}")
    print("Number of samples:", len(df))
    print("Label distribution:")
    print(df["stance"].value_counts())

    print("\nFirst example:")
    print("Text:", df.iloc[0]["comment_text"])
    print("Label:", df.iloc[0]["stance"])
    print("Token IDs:", tokens["input_ids"][0])

    return df, tokens

train_df, train_tokens = prepare_data("train.csv")
val_df, val_tokens = prepare_data("validation.csv")

print("\nTokenization completed successfully!")
