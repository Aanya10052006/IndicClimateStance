from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "FacebookAI/xlm-roberta-base"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pretrained transformer...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={
        0: "Against",
        1: "Favour",
        2: "None"
    },
    label2id={
        "Against": 0,
        "Favour": 1,
        "None": 2
    }
)

print("\nModel loaded successfully!")

# Test tokenization using a Hindi comment
text = "जलवायु परिवर्तन एक गंभीर समस्या है।"

tokens = tokenizer(
    text,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print("\nOriginal text:", text)
print("Token IDs:", tokens["input_ids"])
print("Attention mask:", tokens["attention_mask"])
print("\nNumber of classes:", model.config.num_labels)
