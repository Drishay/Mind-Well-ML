# 🧠 Mind-Well ML  
### Transformer-Based Psychological State Classification System

---

## 📌 Overview

**Mind-Well ML** is a deep learning-based Natural Language Processing (NLP) system designed to classify user-generated text into psychological states using a fine-tuned Transformer model.

The project leverages **DistilBERT (HuggingFace Transformers)** to perform multi-class classification for mental health detection.

This system demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation, visualization, and deployment-ready inference.

---

## 🎯 Problem Statement

Mental health detection from text is a complex contextual problem.  
Traditional machine learning approaches struggle to capture emotional nuance and semantic meaning.

This project aims to build a context-aware Transformer-based classifier capable of identifying psychological categories from raw user text.

---

## 🧠 Target Classes

The model classifies text into 7 psychological states:

- Anxiety  
- Depression  
- Stress  
- Suicidal  
- Bipolar  
- Personality Disorder  
- Normal  

---

## 🏗 Model Architecture

- Base Model: `distilbert-base-uncased`
- Framework: HuggingFace Transformers
- Task Type: Single-label Multi-class Classification
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Fine-tuning Strategy: Supervised Learning

---

## 📊 Dataset Processing

- CSV dataset loaded using Pandas
- Columns renamed (`statement → text`, `status → label`)
- Label encoding using `LabelEncoder`
- Stratified train-validation split (80/20)

---

## 🔄 Text Preprocessing

- Tokenization using `DistilBertTokenizerFast`
- Truncation & Padding (`max_length=128`)
- Custom PyTorch Dataset implementation

---

## 🧪 Training Configuration

- Epochs: 3
- Learning Rate: 2e-5
- Batch Size: 16
- Weight Decay: 0.01
- Evaluation Strategy: Per epoch

---

## 📈 Model Performance

| Metric | Score |
|--------|--------|
| Validation Accuracy | ~82–83% |
| Weighted F1 Score | ~0.82 |

### Evaluation Methods:
- Accuracy
- Precision
- Recall
- Weighted F1 Score
- Confusion Matrix
- ROC Curve (Micro-Average)

---

## 📊 Example Inference

```python
text = "I feel anxious and overwhelmed about my future."
prediction = model.predict(text)
print(prediction)
