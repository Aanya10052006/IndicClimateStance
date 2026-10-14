
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
    keep_default_na=False,
)
test_df = pd.read_csv(
    DATA_DIR / "test.csv",
    encoding="utf-8-sig",
    keep_default_na=False,
)

for name, df in [("training", train_df), ("testing", test_df)]:
    if df["comment_text"].str.strip().eq("").any():
        raise SystemExit(f"Empty comment found in {name} data.")
    if df["stance"].str.strip().eq("").any():
        raise SystemExit(f"Missing stance label in {name} data.")

X_train = train_df["comment_text"].astype(str)
y_train = train_df["stance"].astype(str)
X_test = test_df["comment_text"].astype(str)
y_test = test_df["stance"].astype(str)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents=None,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("Training TF-IDF + Logistic Regression...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

labels = ["Against", "Favour", "None"]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Macro F1:",
    round(f1_score(y_test, predictions, labels=labels,
                   average="macro", zero_division=0), 4),
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )
)

print("\nConfusion matrix (rows=true, columns=predicted)")
print("Label order:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0,
)

with open(RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions, labels=labels,
                average="macro", zero_division=0
            ),
            "classification_report": report,
            "labels": labels,
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=labels
            ).tolist(),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

test_output = test_df.copy()
test_output["predicted_stance"] = predictions
test_output.to_csv(
    RESULTS_DIR / "baseline_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

joblib.dump(model, MODEL_DIR / "tfidf_logistic_baseline.joblib")

print("\nSaved:")
print(" - results/baseline_metrics.json")
print(" - results/baseline_predictions.csv")
print(" - models/tfidf_logistic_baseline.joblib")