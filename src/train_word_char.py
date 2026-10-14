
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
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

X_train = train_df["comment_text"].fillna("").astype(str)
y_train = train_df["stance"].astype(str)
X_test = test_df["comment_text"].fillna("").astype(str)
y_test = test_df["stance"].astype(str)

print("\n--- DATA CHECK ---")
print("Test rows:", len(test_df))
print("Test labels:", len(y_test))
print("Actual label counts:")
print(y_test.value_counts(dropna=False).to_string())
print("Unique labels:", sorted(y_test.unique()))

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents=None,
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=30000,
            min_df=2,
            sublinear_tf=True,
        ),
    ),
])

model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("Training word + character TF-IDF model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Prediction count:", len(predictions))
print("Predicted label counts:")
print(pd.Series(predictions).value_counts().to_string())

labels = ["Against", "Favour", "None"]
macro_f1 = f1_score(
    y_test, predictions, labels=labels,
    average="macro", zero_division=0
)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("\nClassification report:")
print(classification_report(
    y_test, predictions, labels=labels, zero_division=0
))
print("\nConfusion matrix (rows=true, columns=predicted)")
print("Label order:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "word_char_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=labels,
            output_dict=True, zero_division=0
        ),
    }, f, indent=2, ensure_ascii=False)

output = test_df.copy()
output["predicted_stance"] = predictions
output.to_csv(
    RESULTS_DIR / "word_char_predictions.csv",
    index=False, encoding="utf-8-sig"
)

joblib.dump(model, MODEL_DIR / "word_char_logistic.joblib")

print("\nSaved:")
print(" - results/word_char_metrics.json")
print(" - results/word_char_predictions.csv")
print(" - models/word_char_logistic.joblib")