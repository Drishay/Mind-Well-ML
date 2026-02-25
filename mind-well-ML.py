# ============================================================
# 🧠 Mind-Well ML
# Transformer-Based Psychological State Classification
# ============================================================

# =========================
# 1️⃣ Install Dependencies
# =========================

# pip install -r requirements.txt

# =========================
# 2️⃣ Import Libraries
# =========================

import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================
# 3️⃣ Load and Prepare Data
# =========================

df = pd.read_csv("mental-health.csv", encoding="latin-1")

df = df.dropna(subset=["label"])
df = df.reset_index(drop=True)

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

print("Label Mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Stratified Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].astype(str).tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# 4️⃣ Tokenization
# =========================

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

# =========================
# 5️⃣ Create PyTorch Dataset
# =========================

class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MentalHealthDataset(train_encodings, train_labels)
val_dataset = MentalHealthDataset(val_encodings, val_labels)

# =========================
# 6️⃣ Load Model
# =========================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# =========================
# 7️⃣ Training Configuration
# =========================

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# =========================
# 8️⃣ Metrics Function
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# =========================
# 9️⃣ Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# =========================
# 🔟 Train Model
# =========================

trainer.train()

# =========================
# 1️⃣1️⃣ Evaluation
# =========================

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

predictions = trainer.predict(val_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# =========================
# 1️⃣2️⃣ Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Mind-Well ML")
plt.show()

# =========================
# 1️⃣3️⃣ ROC Curve (Micro-Average)
# =========================

logits = predictions.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

n_classes = len(le.classes_)
y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2,
         label=f"Micro-average ROC (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Mind-Well ML")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# =========================
# 1️⃣4️⃣ Save Model
# =========================

trainer.save_model("mind_well_ml_model")
tokenizer.save_pretrained("mind_well_ml_model")

print("Model Saved Successfully.")