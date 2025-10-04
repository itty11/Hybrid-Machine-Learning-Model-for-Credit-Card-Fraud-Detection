# Hybrid-Machine-Learning-Model-for-Credit-Card-Fraud-Detection

This project implements a hybrid ensemble learning approach for detecting fraudulent credit card transactions using the SecurePay Credit Card Fraud Detection Dataset.
The system combines multiple machine learning models — Logistic Regression, Random Forest, and XGBoost — in both Voting and Stacking configurations to improve fraud detection accuracy in highly imbalanced datasets.

# Dataset Details

| Property               | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| **Dataset Name**       | SecurePay: Credit Card Fraud Detection Data                        |
| **Shape**              | (284,807, 31)                                                      |
| **Target Column**      | `Class` — 0 = Normal Transaction, 1 = Fraudulent Transaction       |
| **Fraud Distribution** | Normal (0): 99.82%, Fraud (1): 0.17%                               |
| **Features**           | 30 total — `Time`, `Amount`, `V1`–`V28` (PCA-transformed features) |


Objective: Predict whether a given transaction is fraudulent based on anonymized and preprocessed financial features.

# Machine Learning Pipeline

A. Data Preprocessing

Load and inspect dataset.

Handle missing values (none found).

B. Models Used

| Model               | Type         | Description                                          |
| ------------------- | ------------ | ---------------------------------------------------- |
| Logistic Regression | Linear Model | Baseline for comparison.                             |
| Random Forest       | Ensemble     | Robust to imbalance, captures nonlinearities.        |
| XGBoost             | Boosting     | Handles imbalance efficiently with scale_pos_weight. |
| Voting Classifier   | Hybrid       | Combines LR + RF + XGB with **majority voting**.     |
| Stacking Classifier | Hybrid       | Uses LR as **meta-learner** over base learners.      |


Evaluation Metrics

| Metric                            | Description                                   |
| --------------------------------- | --------------------------------------------- |
| **Accuracy**                      | Overall correctness of predictions            |
| **Precision / Recall / F1-score** | Key metrics for fraud detection               |
| **ROC-AUC**                       | Measures overall model discrimination ability |


Model Performance Summary

| Model                 | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC    |
| --------------------- | -------- | ------------- | ---------- | ------ | ---------- |
| Logistic Regression   | 0.9755   | 0.0609        | 0.9184     | 0.1141 | **0.9721** |
| Random Forest         | 0.9995   | 0.9605        | 0.7449     | 0.8391 | **0.9529** |
| XGBoost               | 0.9994   | 0.8137        | 0.8469     | 0.8300 | **0.9815** |
| **Voting (Hybrid)**   | 0.9994   | 0.7944        | 0.8673     | 0.8293 | **0.9723** |
| **Stacking (Hybrid)** | 0.9976   | 0.4055        | 0.8980     | 0.5587 | **0.9848** |


All trained models and scaler saved inside the models/ directory.

# Technologies Used

Python 3.11+

Libraries:

pandas, numpy, scikit-learn, xgboost, imblearn, joblib, streamlit, matplotlib, seaborn

Streamlit Application

You can interactively test your hybrid fraud detection model via a user-friendly Streamlit app.

# Run the App:

Standardize numerical features using StandardScaler.

Split data into train/test (80/20).

streamlit run app.py

Web Interface:

    Input: Time, Amount, V1–V28
    
    Models: Voting + Stacking
    
    Output: Predicted class (✅ Normal or 🚨 Fraud)
    
    Optional: Add fraud probability score visualization.

Example Prediction Input

| Feature | Example Value                                      |
| ------- | -------------------------------------------------- |
| Time    | 45000                                              |
| Amount  | 120.75                                             |
| V1–V28  | Random PCA feature values (e.g., -1.23, 0.85, ...) |


# Key Learnings

Handling highly imbalanced data using ensemble learning.

Comparison of bagging (RF) vs boosting (XGBoost).

Building hybrid systems with Voting and Stacking.

Deployment-ready ML workflow using Streamlit.

# Next Steps

Add fraud probability display with confidence levels.

Deploy via Streamlit Cloud or Hugging Face Spaces.

Integrate a real-time transaction monitoring API.

# Author

Ittyavira C Abraham

MCA AI Student, Amrita Vishwa Vidyapeetham

Focus: AI/ML, Fraud Detection, and Predictive Systems
