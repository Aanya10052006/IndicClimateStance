from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final"
RESULTS_DIR = ROOT / "results"

LABEL2ID = {
    "Against": 0,
    "Favour": 1,
    "None": 2,
}
LABEL_NAMES = ["Against", "Favour", "None"]
LABEL_IDS = [0, 1, 2]

MAX_LENGTH = 128


def evaluate(model_path, split="validation", weighted=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if split not in ("validation", "test"):
        raise ValueError("Split must be validation or test")

    dataset_path = DATA_DIR / f"{split}.csv"

    df = pd.read_csv(
        dataset_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required = {"comment_text", "stance"}
    if not required.issubset(df.columns):
        raise ValueError("Required columns missing")

    df = df[["comment_text", "stance"]].copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df["stance"] = df["stance"].astype(str).str.strip()

    if not df["stance"].isin(LABEL2ID).all():
        raise ValueError("Invalid stance labels found")

    if df["comment_text"].eq("").any():
        raise ValueError("Empty comments found")

    true_labels = df["stance"].map(LABEL2ID).to_numpy()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading model:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for start in range(0, len(df), 8):
            texts = df["comment_text"].iloc[
                start:start + 8
            ].tolist()

            inputs = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)
            batch_preds = torch.argmax(
                outputs.logits,
                dim=-1
            )

            predictions.extend(
                batch_preds.cpu().tolist()
            )

    predictions = np.array(predictions)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=LABEL_IDS,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=LABEL_IDS,
    )

    metrics = {
        "evaluation_split": split,
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(macro_f1),
        "f1_micro": float(
            f1_score(
                true_labels,
                predictions,
                labels=LABEL_IDS,
                average="micro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                true_labels,
                predictions,
                labels=LABEL_IDS,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predicted_distribution": {
            name: int(np.sum(predictions == i))
            for i, name in enumerate(LABEL_NAMES)
        },
    }

    prefix = "muril_weighted" if weighted else "muril"

    # Keep held-out test outputs separate from validation.
    suffix = "" if split == "validation" else "_test"

    metrics_path = (
        RESULTS_DIR / f"{prefix}{suffix}_metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title(
        f"{prefix} - {split} Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        f"{prefix}{suffix}_confusion_matrix.png",
        dpi=200,
    )

    plt.close(fig)

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=LABEL_IDS,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("\nPredicted Distribution:")
    print(metrics["predicted_distribution"])

    print("\nMetrics saved:", metrics_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--weighted",
        action="store_true",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        split=args.split,
        weighted=args.weighted,
    )