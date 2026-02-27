# 🧠 Building Mind-Well ML  
## The Journey of My First AI System

---

## 🌱 Where It Started

This project didn’t start with code.

It started with a question:

> Can I build an AI that understands psychological tone from text?

Initially, I imagined a full conversational AI system that could:
- Detect emotional state
- Identify suicide/self-harm risk
- Score confidence
- Generate responses

I had no clear pipeline.
Just an idea.

So I opened ChatGPT and started asking:
- What dataset should I use?
- How do I build emotion classification?
- What is the best model for NLP?

That’s how the journey began.

---

## 🧠 The First Plan (Which Was Too Big)

My original vision:

User → Preprocessing → Emotion Model → Risk Detector → LLM → Feedback Loop

But I quickly realized:

> I don’t need a full system.
> I need to build one working component first.

So I simplified the problem.

Version 1 goal:
Build a multi-class psychological text classifier.

That decision made everything manageable.

---

## 📊 Dataset Confusion Phase

I researched multiple datasets:

- GoEmotions (58 emotions, multi-label)
- CLPsych (depression detection)
- Kaggle mental health datasets

At first, I wanted to use GoEmotions.

But then I realized:
- Multi-label classification adds complexity.
- I haven’t even trained a transformer yet.

So I chose a structured 7-class mental health dataset instead.

Lesson learned:
> Scope control is critical in AI projects.

---

## 🤖 Classical ML vs Transformers

My first thought was:
“Should I start with Logistic Regression or SVM?”

But then I thought deeper.

Mental health classification depends on context.

Example:
"I feel hopeless"
vs
"I lost hope in the exam"

Keyword-based models might fail here.

So I decided to directly fine-tune:
`distilbert-base-uncased`

This was my first time using a Transformer model properly.

I didn’t fully understand how it worked internally.
But I committed to learning by building.

---

## ⚙️ The First Training Run

Configuration:
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16

When I saw:
80% accuracy on first epoch

I felt like it was magic.

But then I learned something important:

Validation loss matters more than excitement.

At epoch 3, validation loss slightly increased.

That was my first real encounter with overfitting.

---

## 🛠 Debugging Moments That Taught Me More Than Training

### 1️⃣ GPU / CPU Tensor Error

RuntimeError: Expected all tensors to be on same device.

I had no idea what that meant at first.

Then I learned:
Models and tensors must be on the same device.

That moment made me understand PyTorch deeper.

---

### 2️⃣ Epoch Confusion

I changed epochs from 3 to 2.
Ran training again.
Still saw 3-epoch behavior.

Why?

Because I was continuing training from saved weights.

That’s when I understood:

> Training arguments do not reset the model.
> You must reload base weights to retrain cleanly.

That was a big engineering lesson.

---

### 3️⃣ Colab Runtime Reset

Closed the notebook.
Came back.
Model gone.

That’s when I learned:
Always save models to persistent storage.

Now I understand reproducibility matters.

---

### 4️⃣ Multi-Class ROC Confusion

I wanted to add ROC curve.

But ROC is straightforward only for binary classification.

I learned about:
- One-vs-Rest strategy
- Label binarization
- Micro-average ROC

That was my first exposure to deeper evaluation concepts.

---

## 📈 Final Results

After full training:

- Validation Accuracy ≈ 82–83%
- Weighted F1 ≈ 0.82

Not 95%.
Not perfect.

But it was real.
It was mine.
And I understood every part of it.

---

## 🧠 What Changed in My Thinking

Before this project:
- I thought AI = model + accuracy

After this project:
- AI = data + preprocessing + evaluation + debugging + reproducibility

I now understand:

- Why stratified splits matter
- Why accuracy alone is misleading
- Why overfitting happens
- Why Transformers outperform classical models in context-heavy tasks

---

## 🔮 What I Would Do Differently Now

If I rebuild Mind-Well ML:

- Add class-weighted loss
- Use early stopping
- Compare with classical ML baseline
- Try RoBERTa or DeBERTa
- Implement multi-label version using GoEmotions

But I intentionally kept Version 1 simple.

Because this project wasn’t about building a perfect model.

It was about learning how to build an AI system.

---

## 🏁 Final Reflection

Mind-Well ML is my first end-to-end NLP system.

It marks the transition from:
“Watching tutorials”
to
“Building real AI systems.”

It taught me:
- Engineering discipline
- Experimental control
- Debugging resilience
- Evaluation rigor

And most importantly:

> AI is not magic.
> It’s systematic problem solving.

And that’s how I trained my first real machine learning model — from raw data to a working Transformer-based system.