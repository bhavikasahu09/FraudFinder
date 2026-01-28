# Insurance Fraud Detection (Imbalanced Classification)

This folder contains the complete workflow for an insurance fraud detection project using a highly imbalanced dataset (~6% fraud cases).

## Folder Structure
- `1_eda.ipynb`  
  Exploratory Data Analysis: class imbalance, feature inspection, and data understanding.

- `2_model.ipynb`  
  Model building and evaluation using:
  - Class-weighted models
  - SMOTE oversampling
  - Random Forest
  - Threshold tuning for fraud recall

## Target Variable
- `FraudFound_P` (1 = Fraud, 0 = Non-Fraud)

## Key Focus
- Handling class imbalance
- Optimizing recall, F1-score, and ROC-AUC
- Avoiding accuracy as a misleading metric

## Tools & Libraries
Python, pandas, scikit-learn, imbalanced-learn

## Notes
* This project prioritizes fraud detection performance and realistic evaluation over raw accuracy.
* Class imbalance was handled using cost-sensitive learning and threshold tuning rather than undersampling to avoid information loss.

