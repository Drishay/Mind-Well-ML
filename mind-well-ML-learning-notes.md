# 🧠 Mind-Well ML – Learning Notes
## Quick Recall Notes (For Interviews & Self-Revision)

---

# 1️⃣ What Did I Build?

I built a Transformer-based NLP system that classifies user text into 7 psychological states:

- Anxiety  
- Depression  
- Stress  
- Suicidal  
- Bipolar  
- Personality Disorder  
- Normal  

It is a **single-label multi-class classification problem**.

It is not a clinical diagnostic tool.  
It is a research-based emotional tone classifier.

---

# 2️⃣ What Type of ML Problem Is This?

- Supervised Learning  
- Multi-class Classification  
- Single-label Prediction  
- Text Classification (NLP)

---

# 3️⃣ Why Did I Use DistilBERT?

Because mental health classification depends heavily on context.

Example:
“I feel hopeless”
vs
“I lost hope in the exam”

Traditional ML models struggle with contextual meaning.

DistilBERT:
- Captures semantic context
- Is lighter than full BERT
- Fine-tunes well on small-to-medium datasets
- Balances performance and speed

Model used:
`distilbert-base-uncased`

---

# 4️⃣ Libraries Used and Why

## 🔹 transformers
Used for:
- DistilBERT model
- Tokenizer
- Trainer API

Why?
- Provides state-of-the-art NLP models
- Simplifies fine-tuning
- Handles training loop abstraction

Without transformers, I would need to manually implement:
- Attention masks
- Optimizer scheduling
- Training loop logic

---

## 🔹 torch (PyTorch)
Used for:
- Tensor computation
- GPU acceleration
- Custom Dataset class

Why?
- Transformers library runs on PyTorch
- Enables hardware acceleration
- Provides Dataset abstraction for training

---

## 🔹 scikit-learn
Used for:
- LabelEncoder
- train_test_split (stratified)
- Accuracy / F1-score
- Confusion matrix
- ROC curve

Why?
- Reliable evaluation utilities
- Standard ML metric toolkit

---

## 🔹 pandas
Used for:
- Loading CSV
- Cleaning dataset
- Renaming columns
- Handling missing values

Why?
- Structured and efficient data manipulation

---

## 🔹 numpy
Used for:
- Argmax for predictions
- Numerical operations
- Metric calculations

Why?
- Core numerical backbone for ML workflows

---

## 🔹 matplotlib & seaborn
Used for:
- Confusion matrix visualization
- ROC curve plotting
- Accuracy graphs

Why?
- Helps visually interpret model performance

---

## 🔹 datasets (HuggingFace) – Future Upgrade
Not fully used in Version 1.

Planned for:
- Loading GoEmotions dataset
- Scalable dataset handling

---

# 5️⃣ Data Pipeline Summary

1. Load CSV
2. Remove null labels
3. Encode labels using LabelEncoder
4. Stratified 80/20 train-validation split
5. Tokenize using DistilBERT tokenizer
6. Convert to PyTorch Dataset
7. Train using HuggingFace Trainer

---

# 6️⃣ Training Configuration

- Epochs: 3
- Learning Rate: 2e-5
- Batch Size: 16
- Weight Decay: 0.01
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW

Why 2e-5 learning rate?
→ Standard safe value for transformer fine-tuning to avoid catastrophic forgetting.

---

# 7️⃣ Evaluation Metrics Used

- Accuracy
- Weighted F1-score
- Precision
- Recall
- Confusion Matrix
- Micro-average ROC curve

Why weighted F1?
→ Dataset is slightly imbalanced. Accuracy alone is misleading.

---

# 8️⃣ Final Performance

- Validation Accuracy ≈ 82–83%
- Weighted F1 ≈ 0.82

Observed:
- Slight validation loss increase at epoch 3
- Early overfitting signal

---

# 9️⃣ Challenges Faced

## GPU / CPU Tensor Mismatch
Resolved by moving model and inputs to same device.

## Epoch Confusion
Training continued from saved weights.
Learned that model must be reloaded to retrain cleanly.

## Colab Runtime Reset
Solved by saving model to Google Drive.

## Multi-class ROC
Learned that ROC for multi-class requires label binarization.

---

# 🔟 What I Learned Technically

- Transformer fine-tuning workflow
- Importance of validation loss
- Stratified splitting matters
- Accuracy is not enough
- Model persistence is critical
- Trainer API abstracts training loop but understanding internals is important

---

# 1️⃣1️⃣ What Would I Improve?

- Add class-weighted loss
- Use EarlyStopping
- Try RoBERTa / DeBERTa
- Compare against classical ML baseline
- Convert to multi-label using GoEmotions

---

# 1️⃣2️⃣ 30-Second Explanation (Interview Version)

“I built a transformer-based NLP classifier using DistilBERT to detect psychological states from text. I fine-tuned it on a labeled dataset, achieved about 83% validation accuracy, and evaluated it using weighted F1 and ROC curves. I handled GPU training, class imbalance considerations, and model persistence for reproducibility.”

---

# 🏁 Final Takeaway

This project taught me:

AI is not just about accuracy.
It is about designing, training, evaluating, debugging, and improving a system systematically.

This was my first real end-to-end deep learning NLP system.