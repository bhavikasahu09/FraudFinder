import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================
# Load assets
# ================

import os
import joblib

# 1. Get the absolute path to the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with the relative path to the models folder
# This goes up one level from 'app' to the root, then into 'models'
models_path = os.path.join(current_dir, "..", "models")

# 3. Load your assets using the joined paths
model = joblib.load(os.path.join(models_path, "best_xgb_model.joblib"))
encoder = joblib.load(os.path.join(models_path, "encoder.joblib"))
scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))
feature_names = joblib.load(os.path.join(models_path, "feature_names.joblib"))

X_val = joblib.load(os.path.join(models_path, "X_val.joblib"))
y_val = joblib.load(os.path.join(models_path, "y_val.joblib"))

threshold = 0.14

# ====================
# Dynamic metrics
# ====================
y_prob = model.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)


# =========
# UI
# =========
st.title("🛡️ FraudFinder – Insurance Claim Risk")

st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall (Fraud)", f"{recall:.2%}")
c2.metric("Precision (Fraud)", f"{precision:.2%}")
c3.metric("F1 Score", f"{f1:.2f}")
c4.metric("Accuracy", f"{accuracy:.2%}")
st.divider()

st.sidebar.header("Claim Details")

fault = st.sidebar.radio("Fault", ["Policy Holder", "Third Party"])
base_policy = st.sidebar.selectbox(
    "Base Policy", ["All Perils", "Collision", "Liability"]
)
vehicle_cat = st.sidebar.selectbox("Vehicle Category", ["Sedan", "Sport", "Utility"])
vehicle_price = st.sidebar.selectbox(
    "Vehicle Price",
    [
        "less than 20000",
        "20000 to 29000",
        "30000 to 39000",
        "40000 to 59000",
        "60000 to 69000",
        "more than 69000",
    ],
)
past_claims = st.sidebar.selectbox(
    "Past Claims", ["none", "1", "2 to 4", "more than 4"]
)

month = st.selectbox(
    "Month",
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
)
age = st.number_input("Age", 18, 100, 30)

# =============
# Prediction
# =============
if st.button("Analyze Risk"):

    # ======================
    # Build user input
    # ======================
    user_data = pd.DataFrame(
        [
            {
                "Fault": fault,
                "BasePolicy": base_policy,
                "VehicleCategory": vehicle_cat,
                "VehiclePrice": vehicle_price,
                "PastNumberOfClaims": past_claims,
                "Month": month,
                "Age": age,
            }
        ]
    )

    # ===============================
    # Align with training columns
    # ===============================
    cat_cols = list(encoder.feature_names_in_)
    num_cols = [c for c in feature_names if c not in encoder.get_feature_names_out()]

    # Add missing categorical columns
    for col in cat_cols:
        if col not in user_data.columns:
            user_data[col] = "Unknown"

    # Add missing numerical columns
    for col in num_cols:
        if col not in user_data.columns:
            user_data[col] = 0

    # Ensure correct order
    user_data = user_data[cat_cols + num_cols]

    # ================
    # Encode + scale
    # ================
    # Encode categorical features and scale numerical features
    # so the user input matches the training data format
    X_cat = encoder.transform(user_data[cat_cols])
    X_num = user_data[num_cols].values

    X_final = np.hstack([X_num, X_cat])
    X_final = scaler.transform(X_final)

    prob = model.predict_proba(X_final)[0][1]

    # Classify risk based on the pre-defined threshold
    if prob >= threshold:
        st.error(f"🚨 HIGH RISK — {prob:.2%}")
    else:
        st.success(f"✅ LOW RISK — {prob:.2%}")

    # ==================
    # SHAP Explanation
    # ==================

    # Explain model prediction using SHAP for transparency
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final)

    shap_df = pd.DataFrame(
        {"Feature": feature_names, "Impact": shap_values[0]}
    ).sort_values(by="Impact", key=abs, ascending=False)

    st.subheader("🔍 Why this decision?")
    st.dataframe(shap_df.head(6))
