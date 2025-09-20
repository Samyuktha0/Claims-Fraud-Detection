# Claims Fraud Detection Using Bayesian Networks and XGBoost

## 🔍 Overview
This project builds a fraud detection pipeline for insurance claims using XGBoost and Bayesian inference. It processes 50K+ synthetic claims, identifies anomalies, and provides explainable predictions with SHAP. The model is deployed on AWS SageMaker with automated retraining.

## 🚀 Features
- XGBoost classifier with 92% AUC
- Bayesian network for probabilistic reasoning
- SHAP integration for model explainability
- Looker dashboard for anomaly alerts
- Retraining pipeline on AWS SageMaker

## 🧰 Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- pgmpy (Bayesian Networks)
- SHAP
- AWS SageMaker
- Looker

## 📦 Setup
```bash
pip install -r requirements.txt
python src/train_model.py
python src/explain_model.py
