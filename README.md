# 🛡️ FraudFinder: Insurance Claim Risk Prediction

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://fraudfinder-insurance-claim-detection.streamlit.app/)

**FraudFinder** is an end-to-end machine learning pipeline designed to mitigate financial losses in the **BFSI (Banking, Financial Services, and Insurance)** sector. By automating the initial screening of insurance claims, it allows adjusters to focus on high-probability fraud cases, significantly reducing fraud cases 

## 🌐 Live Demo
👉 **Streamlit App**: [ WebApp ](https://fraudfinder-insurance-claim-detection.streamlit.app/)

It uses a tuned **XGBoost classifier** to flag high-risk claims and provides model interpretability using **SHAP values**.

---

## 📊 Model Strategy & Performance

Insurance fraud detection is a highly imbalanced classification problem. To reduce missed fraud cases, the model is optimized primarily for **Recall**.

### Model Details Explained

The system is powered by an **XGBoost (Extreme Gradient Boosting)** classifier, chosen for its superior performance with structured tabular data and its native ability to handle missing values and outliers.

* **Iterative Learning**: The XGBoost engine builds trees sequentially, where each new tree specifically addresses the residual errors of the previous ones to create a highly accurate ensemble.
* **Handling Class Imbalance**: Fraudulent claims typically represent less than 10% of total data. To prevent the model from ignoring the minority class, we utilized `scale_pos_weight` to increase the penalty for misclassifying fraud.
* **Optimization Strategy**: We optimized the model for **Recall** rather than Accuracy to ensure that as many fraudulent claims as possible are flagged. Through a precision-recall trade-off analysis, a custom probability threshold of **0.14** was implemented.


### Details
* **Algorithm**: XGBoost Classifier
* **Class Imbalance Handling**: `scale_pos_weight = 15.714092140921409`
* **Custom Decision Threshold**: `0.14`

### Key Metrics (Fraud Class)

* **Recall**: ~93.51%
* **Precision**: ~13.01%
* **F1-Score**: ~0.23

**Metric Priority**: In the BFSI domain, we prioritize Recall over Precision. It is more critical to catch every potential risk (minimizing False Negatives) than to avoid occasional false alarms.

---

## 🔍 Explainability with SHAP

The application integrates **SHAP (SHapley Additive exPlanations)** to explain individual predictions. For each claim, the dashboard highlights the most influential features contributing to the fraud risk score, such as:

* `Age`
* `PastNumberOfClaims`
* `VehiclePrice`

### Pros
* **Feature Contribution**: SHAP values quantify the exact "pull" of each feature (e.g., `Age`, `VehiclePrice`, or `PastNumberOfClaims`) toward a specific risk score.
* **Local Interpretability**: For every "High Risk" flag, the dashboard displays a breakdown of the primary drivers, allowing human auditors to justify why a specific claim was marked for investigation.

This improves transparency and trust in model decisions.

---

## 🛠️ Tech Stack

* **Language**: Python 3.8+
* **Machine Learning**: XGBoost, Scikit-learn, SHAP
* **Frontend**: Streamlit
* **Data Processing**: Pandas, NumPy

---

## 🚀 Setup and Usage

### 1. Installation

```bash
git clone https://github.com/sumitnagpure/BDT.git
cd FraudFinder
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app/app.py
```

---

## 📂 Repository Structure

```
FraudFinder/
│
├── app/         # Streamlit web application
├── models/      # Trained model and preprocessing objects (joblib)
├── notebooks/   # EDA and model training notebooks
├── data/        # Raw and processed datasets
└── requirements.txt
```


> You can interact with the deployed model and SHAP explanations via the Streamlit dashboard.
> ➡ [ WebApp ](https://fraudfinder-insurance-claim-detection.streamlit.app/)


---

## 📌 Notes

* The model is recall-focused and may produce higher false positives.
* Threshold tuning is intentionally aggressive to minimize missed fraud cases.
* SHAP plots are generated dynamically for prediction-level explainability.
