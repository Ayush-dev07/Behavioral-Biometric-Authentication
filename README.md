# 🧠 Behavioral Biometric Authentication
### Keystroke Dynamics using Siamese RNN with Triplet Loss

A deep learning–based behavioral biometric authentication system that verifies users based on **typing patterns** rather than static passwords.

This project implements a **Siamese RNN architecture trained with Triplet Loss**, enabling robust user verification, continuous authentication, and anti-spoofing capabilities.

---

## 🚀 Project Overview

Traditional authentication systems rely on *what you know* (passwords).  
This system focuses on *how you type*.

It analyzes:

- Key **dwell time** (press duration)
- **Flight time** (time between key presses)
- Digraph timing patterns
- Typing rhythm variability

The model learns an embedding space where:

- Sequences from the same user cluster together
- Sequences from different users are separated by a margin

---

## 🏗️ System Architecture

Keyboard Events → Feature Extraction → RNN Encoder  
→ Embedding Space → Distance Computation → Authentication Decision

### Core Model

- Shared-weight GRU/LSTM encoder
- Projection layer
- L2-normalized embedding
- Triplet Loss (margin-based metric learning)

---

## 🔬 Key Features

- Siamese RNN architecture
- Triplet loss with margin separation
- Semi-hard negative mining
- Continuous authentication capability
- Anti-spoofing support
- FAR / FRR / EER evaluation

---

## 📊 Biometrics Metrics

Performance is evaluated using:

- **FAR (False Accept Rate)**
- **FRR (False Reject Rate)**
- **EER (Equal Error Rate)**
- ROC curve analysis

Target benchmark:
EER < 5–10%

