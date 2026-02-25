# 🧠 Mind-Well ML  
### Transformer-Based Psychological State Classification System

Mind-Well ML is a deep learning-based Natural Language Processing (NLP) system that classifies user-generated text into psychological states using a fine-tuned Transformer model (DistilBERT).

This project demonstrates end-to-end model development including data preprocessing, transformer fine-tuning, evaluation, ROC analysis, and model deployment readiness.

---

## 🚀 Project Overview

Mental health detection from text is a critical task in digital health systems. Traditional rule-based systems fail to capture contextual nuances in language.

Mind-Well ML leverages **DistilBERT**, a pre-trained Transformer model, fine-tuned for multi-class psychological state classification.

---

## 🎯 Objective

To classify text input into one of the following psychological categories:

- Anxiety  
- Depression  
- Stress  
- Suicidal  
- Bipolar  
- Personality Disorder  
- Normal  

The goal is to support digital mental health screening systems.

---

## 🏗️ Model Architecture

- **Base Model:** distilbert-base-uncased  
- **Framework:** HuggingFace Transformers + PyTorch  
- **Task Type:** Single-label Multi-class Classification  
- **Number of Classes:** 7  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Fine-Tuning Strategy:** Supervised learning  

---

## 📊 Dataset Processing

- CSV dataset loaded using pandas  
- Columns renamed (`statement → text`, `status → label`)  
- Missing labels removed  
- Label encoding using `LabelEncoder`  
- Stratified 80/20 train-validation split  

---

## 🔄 Text Preprocessing

- Tokenization using `DistilBertTokenizerFast`
- Truncation and padding (`max_length = 128`)
- Conversion into PyTorch Dataset class

---

## 🧪 Training Configuration

- **Epochs:** 3  
- **Learning Rate:** 2e-5  
- **Batch Size:** 16  
- **Weight Decay:** 0.01  
- **Evaluation Strategy:** Per epoch  

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | ~82–83% |
| Weighted F1 Score | ~0.82 |
| Task Type | Multi-class Classification |

### Evaluation Methods Used:
- Accuracy
- Precision
- Recall
- Weighted F1 Score
- Confusion Matrix
- Micro-average ROC Curve

---

## 📊 ROC Curve (Micro-Average)

A micro-averaged ROC curve was generated using a One-vs-Rest strategy for multi-class classification.

This demonstrates strong discriminative ability across psychological classes.

---

## 🧠 Key Learnings

- Transformers outperform traditional ML for contextual NLP tasks.
- Validation loss must be monitored to detect overfitting.
- Accuracy alone is insufficient for imbalanced datasets.
- Multi-class ROC requires label binarization.
- Model persistence is critical in cloud-based environments like Google Colab.

---

## 🧪 Manual Inference Example

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mental_health_model",
    tokenizer="mental_health_model"
)

classifier("I feel anxious and scared about my future.")
```

---

## 🏆 Future Improvements

- Implement class-weighted loss for imbalance handling
- Add early stopping
- Experiment with RoBERTa / DeBERTa
- Convert to multi-label classification
- Deploy as REST API using FastAPI
- Integrate into chatbot interface

---

## 📁 Project Structure

```
Mind-Well-ML/
│
├── mind_well_ML.py
├── README.md
├── mind-well-ML-learning.md
├── mind-well-ML-documentation.md
├── requirements.txt
└── mental-health.csv (dataset)
```

---

## 📌 Author

Developed as part of an AI/ML research initiative focused on mental health analytics.

---

## ⚠️ Disclaimer

This system is intended for research and educational purposes only.  
It is not a clinical diagnostic tool.

---

# 🔥 Mind-Well ML
Building AI for Psychological Insight.
