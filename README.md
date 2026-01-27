# 🛡️ FraudFinder: Insurance Claim Risk Prediction

FraudFinder is a machine learning solution designed to identify fraudulent insurance claims in the BFSI domain. It uses a tuned **XGBoost classifier** to flag high-risk claims and provides model interpretability using **SHAP values**.

---

## 📊 Model Strategy & Performance

Insurance fraud detection is a highly imbalanced classification problem. To reduce missed fraud cases, the model is optimized primarily for **Recall**.

### Model Details

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

---

## 📌 Notes

* The model is recall-focused and may produce higher false positives.
* Threshold tuning is intentionally aggressive to minimize missed fraud cases.
* SHAP plots are generated dynamically for prediction-level explainability.
