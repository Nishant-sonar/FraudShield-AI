# 🛡️ FraudShield AI

### Intelligent AML Fraud Detection System with Explainable AI

> A production-ready fraud detection system combining machine learning, rule-based intelligence, explainable AI, and containerized deployment using Docker.

---

## 🚀 Overview

**FraudShield AI** is an end-to-end Anti-Money Laundering (AML) fraud detection system designed to detect suspicious financial transactions with high recall and real-time decision-making.

This system integrates:

* 🧠 Machine Learning Prediction
* ⚙️ Rule-Based Risk Enhancement
* 🔍 Explainable AI (SHAP)
* 🐳 Containerized Deployment (Docker)
* ☁️ MongoDB Atlas (Cloud Storage)

---

## 🎯 Problem Statement

* High false positives in traditional AML systems
* Missing fraud transactions is costly
* Lack of transparency in ML decisions

👉 Solution: **Hybrid Intelligence System (ML + Rules + Explainability + Storage)**

---

## ✨ Key Features

✅ Random Forest ML Model (High Recall Focus)
✅ Rule-Based Risk Boosting Engine
✅ SHAP Explainability
✅ Risk Scoring System (0–100)
✅ Streamlit Interactive Dashboard
✅ MongoDB Atlas Integration (Cloud Storage)
✅ Auto Transaction Logging
✅ Fault-Tolerant System Design
✅ Docker Containerization

---

## 🧠 Data Science Workflow

### 1. Data Understanding

* Analyzed class imbalance
* Identified fraud patterns

### 2. Data Ingestion

* Loaded structured transaction dataset

### 3. Data Transformation

* Encoding & scaling

### 4. Feature Engineering

* Created model-ready features

### 5. Model Training

* Evaluated multiple models
* Selected **Random Forest**

### 6. Model Evaluation

* Precision
* Recall (primary focus)
* F1 Score

### 7. Prediction Pipeline

* Real-time inference system

### 8. Risk Scoring

* Converts probability → 0–100

### 9. Rule Engine

Boosts detection using:

* High transaction amount
* Currency mismatch
* Suspicious bank IDs
* Cross-border patterns

### 10. Explainability

* SHAP-based interpretation

### 11. Data Storage

* MongoDB Atlas for transaction history

---

## 🏗️ System Architecture

```text
User Input (Streamlit UI)
        ↓
Data Validation & Preprocessing
        ↓
Feature Transformation
        ↓
Machine Learning Model (Random Forest)
        ↓
Fraud Probability
        ↓
Risk Scoring (0–100)
        ↓
Rule-Based Enhancement
        ↓
SHAP Explainability
        ↓
Final Output (UI)

        ↘
         MongoDB Atlas (Cloud Storage)
```

---

## 📁 Project Structure

```
AML-Fraud-Detection-main/
│
├── app_professional_dashboard.py
├── app.py
├── shap_explainer.py
│
├── aml_fraud_detector/
│   ├── components/
│   ├── pipeline/
│   ├── utils/
│   └── rule_engine.py
│
├── artifacts/
├── config/
├── mlruns/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Technology Stack

* **Frontend**: Streamlit
* **Backend**: Flask
* **ML**: Scikit-learn (Random Forest)
* **Explainability**: SHAP
* **Database**: MongoDB Atlas
* **Tracking**: MLflow
* **Containerization**: Docker

---

## 🐳 Docker Deployment 

The entire application is containerized using Docker, ensuring consistent execution across environments.

### Build Image

```bash
docker build -t fraudshield-ai .
```

### Run Container

```bash
docker run -p 8501:8501 fraudshield-ai
```

👉 Access: http://localhost:8501

---

## 🧪 How It Works

1. User inputs transaction
2. System preprocesses data
3. ML model predicts fraud probability
4. Risk score calculated
5. Rule engine enhances detection
6. SHAP explains result
7. Output displayed
8. Transaction stored in MongoDB

---

## ☁️ MongoDB Integration

* Stores transaction + prediction results
* Enables historical tracking
* Fault-tolerant (no app crash if DB fails)

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app_professional_dashboard.py
```

---

## 📊 Model Details

* Algorithm: Random Forest
* Dataset: ~15,000 transactions
* Features: 7
* Focus: **High Recall (Fraud Detection Priority)**

---

## 📊 Risk Levels

| Score  | Level  |
| ------ | ------ |
| 0–30   | SAFE   |
| 31–50  | LOW    |
| 51–75  | MEDIUM |
| 76–100 | HIGH   |

---

## 💡 Key Highlight

👉 This is not just an ML model —
it is a **complete intelligent fraud detection system** combining:

* Data understanding
* Machine learning
* Business rules
* Explainability
* Cloud storage
* Containerized deployment

---

## 📄 License

MIT License

---

## 🚀 Status

✅ Production Ready
