from pathlib import Path
import inspect
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -------------------------
# Configuration
# -------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models" / "muril_weighted"

MODEL_NAME = "google/muril-base-cased"

LABEL2ID = {
    "Against": 0,
    "Favour": 1,
    "None": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SEED = 42
MAX_LENGTH = 128

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Dataset loading
# -------------------------

def load_dataset(filename):
    df = pd.read_csv(
        DATA_DIR / filename,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required = {"comment_text", "stance"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {filename}"
        )

    df = df[["comment_text", "stance"]].copy()

    df["comment_text"] = (
        df["comment_text"].astype(str).str.strip()
    )

    df["stance"] = (
        df["stance"].astype(str).str.strip()
    )

    if not df["stance"].isin(LABEL2ID).all():
        raise ValueError("Invalid stance labels found")

    if df["comment_text"].str.len().eq(0).any():
        raise ValueError("Empty comments found")

    df["labels"] = df["stance"].map(LABEL2ID).astype(int)

    return Dataset.from_pandas(
        df[["comment_text", "labels"]],
        preserve_index=False,
    )


# -------------------------
# Evaluation metrics
# -------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_micro": f1_score(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="micro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="weighted",
            zero_division=0,
        ),
    }


# -------------------------
# Weighted Trainer
# -------------------------

class WeightedTrainer(Trainer):

    def __init__(
        self,
        *args,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if class_weights is None:
            raise ValueError(
                "class_weights must be provided"
            )

        self.class_weights = (
            class_weights.detach().clone()
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs["labels"]

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels"
        }

        outputs = model(**model_inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(
                device=logits.device,
                dtype=torch.float32,
            )
        )

        loss = loss_fn(
            logits.float().view(-1, 3),
            labels.view(-1),
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# -------------------------
# Main training pipeline
# -------------------------

def main():
    set_seed(SEED)

    print("Loading datasets...")

    train_dataset = load_dataset("train.csv")
    val_dataset = load_dataset("validation.csv")

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_labels = np.array(
        train_dataset["labels"]
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_labels,
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print("\nClass weights:")

    for i, name in ID2LABEL.items():
        print(
            f"{name}: {class_weights[i]:.4f}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tokenized = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["comment_text"],
    )

    val_tokenized = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["comment_text"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        seed=SEED,
        data_seed=SEED,
        save_total_limit=2,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_tokenized,
        "eval_dataset": val_tokenized,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "class_weights": class_weights,
    }

    trainer_params = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    print("\nStarting weighted MuRIL training...")

    trainer.train()

    print("\nEvaluating on validation data...")

    metrics = trainer.evaluate()

    with open(
        RESULTS_DIR / "muril_weighted_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=4)

    best_model_dir = MODEL_DIR / "best_model"

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    print("\nWeighted MuRIL experiment completed!")
    print("Metrics:", metrics)
    print("Best model saved to:", best_model_dir)


if __name__ == "__main__":
    main()