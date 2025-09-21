import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
import os

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("data/reviews.csv")
df = df.dropna(subset=["review", "label"])  

# If label is already numeric (0,1,2), just use as-is
label2id = {0: 0, 1: 1, 2: 2}  # map integers to themselves
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[["review", "label"]])

# -----------------------
# Train/test split
# -----------------------
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -----------------------
# Tokenization
# -----------------------
MODEL_NAME = "bert-base-uncased"  # can switch to distilbert-base-uncased for faster training
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["review"], padding="max_length", truncation=True)

tokenized_ds = dataset.map(tokenize, batched=True)

# -----------------------
# Load model
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id={str(k): v for k, v in label2id.items()}
)

# -----------------------
# Metrics
# -----------------------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# -----------------------
# Training
# -----------------------
training_args = TrainingArguments(
    output_dir="outputs/bert_sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------
# Save model & tokenizer
# -----------------------
os.makedirs("outputs/bert_sentiment", exist_ok=True)
trainer.save_model("outputs/bert_sentiment")
tokenizer.save_pretrained("outputs/bert_sentiment")
print("Model and tokenizer saved to outputs/bert_sentiment/")
