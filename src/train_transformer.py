from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final"
MODEL_DIR = ROOT / "models" / "xlm_roberta"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "FacebookAI/xlm-roberta-base"

LABEL2ID = {
    "Against": 0,
    "Favour": 1,
    "None": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_data(filename):
    df = pd.read_csv(
        DATA_DIR / filename,
        encoding="utf-8-sig",
        keep_default_na=False
    )

    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df["stance"] = df["stance"].astype(str).str.strip()

    df = df[
        df["comment_text"].ne("")
        & df["stance"].isin(LABEL2ID)
    ].copy()

    df["labels"] = df["stance"].map(LABEL2ID).astype(int)

    return Dataset.from_pandas(
        df[["comment_text", "labels"]],
        preserve_index=False
    )

print("Loading datasets...")
train_dataset = load_data("train.csv")
val_dataset = load_data("validation.csv")

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["comment_text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_f1 = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )[2]

    micro_f1 = precision_recall_fscore_support(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )[2]

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "f1_micro": micro_f1
    }

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

print("Starting fine-tuning...")
trainer.train()

print("Evaluating...")
metrics = trainer.evaluate()

trainer.save_model(str(MODEL_DIR / "final"))
tokenizer.save_pretrained(str(MODEL_DIR / "final"))

with open(
    RESULTS_DIR / "xlm_roberta_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

print("\nTraining completed!")
print(metrics)
