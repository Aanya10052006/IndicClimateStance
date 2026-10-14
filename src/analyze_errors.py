
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Analyze the word + character TF-IDF model
pred_file = RESULTS / "word_char_predictions.csv"
output_file = RESULTS / "word_char_errors.csv"

df = pd.read_csv(
    pred_file,
    encoding="utf-8-sig",
    keep_default_na=False
)

required = {"comment_text", "stance", "predicted_stance"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

errors = df[df["stance"] != df["predicted_stance"]].copy()

errors["actual_stance"] = errors["stance"]
errors["error_type"] = (
    errors["actual_stance"].astype(str)
    + " -> "
    + errors["predicted_stance"].astype(str)
)

columns = [
    c for c in [
        "comment_id",
        "language",
        "video_id",
        "comment_text",
        "actual_stance",
        "predicted_stance",
        "error_type"
    ]
    if c in errors.columns
]

errors[columns].to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"Prediction file: {pred_file.name}")
print(f"Total test comments: {len(df)}")
print(f"Correct predictions: {len(df) - len(errors)}")
print(f"Misclassified comments: {len(errors)}")

print("\nError types:")
print(errors["error_type"].value_counts().to_string())

print("\nExamples of incorrectly predicted Against comments:")
against_errors = errors[errors["actual_stance"] == "Against"]

if against_errors.empty:
    print("No incorrectly predicted Against comments.")
else:
    for _, row in against_errors.head(10).iterrows():
        print("\nActual:", row["actual_stance"])
        print("Predicted:", row["predicted_stance"])
        print("Comment:", row["comment_text"])

print(f"\nSaved detailed errors to results/{output_file.name}")