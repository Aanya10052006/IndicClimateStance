
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "raw_comments.csv"
SAMPLE_FILES = [
    ROOT / "data" / "processed" / "annotation_sample.csv",
    ROOT / "data" / "processed" / "annotation_sample_2.csv",
]
OUTPUT_FILE = ROOT / "data" / "processed" / "against_candidates.csv"

# Search clues only; a keyword match is NOT an Against label.
KEYWORDS = [
    r"\bfake\s+news\b",
    r"\bfake\s+(?:climate|warming|story|stories|narrative)\b",
    r"\bclimate\s+change\s+(?:is\s+)?(?:fake|a\s+hoax|not\s+real)\b",
    r"\bglobal\s+warming\s+(?:is\s+)?(?:fake|a\s+hoax|not\s+real)\b",
    r"\bhoax\b",
    r"\bpropaganda\b",
    r"\bscam\b",
    r"\bfake\b",
    r"\bfraud\b",
    r"\bjhooth\b|\bjhoota\b|\bjhooti\b|\bbakwas\b|\bfarzi\b|\bdhokha\b",
    r"जलवायु\s*परिवर्तन\s*(?:नहीं|झूठ|फर्जी|बकवास)",
    r"ग्लोबल\s*वार्मिंग\s*(?:नहीं|झूठ|फर्जी|बकवास)",
    r"झूठ|फर्जी|बकवास|धोखा",
    r"মিথ্যা|ভুয়া|ভুয়ো|বানোয়াট|প্রোপাগান্ডা",
    r"জলবায়ু\s*পরিবর্তন\s*(?:নয়|মিথ্যা|ভুয়া|ভুয়ো)",
    r"গ্লোবাল\s*ওয়ার্মিং\s*(?:নয়|মিথ্যা|ভুয়া|ভুয়ো)",
]
KEYWORD_PATTERN = re.compile("|".join(KEYWORDS), flags=re.IGNORECASE)


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


if not RAW_FILE.exists():
    raise SystemExit(f"Raw dataset not found: {RAW_FILE}")

raw = read_csv(RAW_FILE)

required = {"comment_id", "comment_text"}
missing = required - set(raw.columns)
if missing:
    raise SystemExit(f"Missing columns in raw dataset: {sorted(missing)}")

# Exclude the 250 comments already sampled for annotation.
annotated_ids = set()

for file in SAMPLE_FILES:
    if file.exists():
        sample = read_csv(file)
        if "comment_id" in sample.columns:
            annotated_ids.update(sample["comment_id"].astype(str))

raw["comment_id"] = raw["comment_id"].astype(str)
remaining = raw[~raw["comment_id"].isin(annotated_ids)].copy()

# Find candidate comments, but do NOT assign labels automatically.
remaining["candidate_keyword"] = remaining["comment_text"].astype(str).apply(
    lambda text: ", ".join(
        dict.fromkeys(
            match.group(0)
            for match in KEYWORD_PATTERN.finditer(text)
        )
    )
)

candidates = remaining[remaining["candidate_keyword"] != ""].copy()
candidates["stance"] = ""
candidates["annotation_notes"] = ""

preferred_columns = [
    "comment_id",
    "video_id",
    "language",
    "comment_text",
    "candidate_keyword",
    "stance",
    "annotation_notes",
]
output_columns = [col for col in preferred_columns if col in candidates.columns]
candidates = candidates[output_columns]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
candidates.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nCandidate search complete.")
print(f"Total raw comments: {len(raw)}")
print(f"Already sampled: {len(annotated_ids)}")
print(f"Unannotated comments searched: {len(remaining)}")
print(f"Keyword-matched candidates: {len(candidates)}")
print(f"Saved to: {OUTPUT_FILE}")
print("\nReminder: keyword matches are NOT automatic Against labels.")
print("Review each comment and its video context before assigning a stance.")