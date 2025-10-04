import streamlit as st
import pandas as pd
import joblib

# ================================
# Load Saved Models & Artifacts
# ================================
voting_model = joblib.load("models/voting_model.pkl")
stacking_model = joblib.load("models/stacking_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
scaler = joblib.load("models/scaler.pkl")  # ensure this was saved in training

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 SecurePay: Credit Card Fraud Detection (Hybrid ML Model)")
st.markdown("This app uses **Voting** and **Stacking** ensemble models to detect fraudulent transactions.")

# ================================
# Collect User Input
# ================================
st.subheader("🧾 Enter Transaction Details:")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Transaction Time", min_value=0.0, value=0.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    v7 = st.number_input("V7", value=0.0)
    v8 = st.number_input("V8", value=0.0)
    v9 = st.number_input("V9", value=0.0)
    v10 = st.number_input("V10", value=0.0)
    v11 = st.number_input("V11", value=0.0)
    v12 = st.number_input("V12", value=0.0)
    v13 = st.number_input("V13", value=0.0)
    v14 = st.number_input("V14", value=0.0)

with col2:
    v15 = st.number_input("V15", value=0.0)
    v16 = st.number_input("V16", value=0.0)
    v17 = st.number_input("V17", value=0.0)
    v18 = st.number_input("V18", value=0.0)
    v19 = st.number_input("V19", value=0.0)
    v20 = st.number_input("V20", value=0.0)
    v21 = st.number_input("V21", value=0.0)
    v22 = st.number_input("V22", value=0.0)
    v23 = st.number_input("V23", value=0.0)
    v24 = st.number_input("V24", value=0.0)
    v25 = st.number_input("V25", value=0.0)
    v26 = st.number_input("V26", value=0.0)
    v27 = st.number_input("V27", value=0.0)
    v28 = st.number_input("V28", value=0.0)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)

# Prepare input dictionary
test_input = {
    "Time": time,
    "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6, "V7": v7,
    "V8": v8, "V9": v9, "V10": v10, "V11": v11, "V12": v12, "V13": v13, "V14": v14,
    "V15": v15, "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20, "V21": v21,
    "V22": v22, "V23": v23, "V24": v24, "V25": v25, "V26": v26, "V27": v27, "V28": v28,
    "Amount": amount
}

# ================================
# Prediction Section
# ================================
if st.button("🔍 Predict Fraud"):
    try:
        # Create DataFrame
        input_df = pd.DataFrame([test_input])

        # Match training feature order
        input_df = input_df[feature_names]

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict with both models
        pred_voting = voting_model.predict(input_scaled)[0]
        pred_stacking = stacking_model.predict(input_scaled)[0]

        st.markdown("---")
        st.subheader("🧠 Model Predictions")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Voting Model", "🚨 Fraud" if pred_voting == 1 else "✅ Normal Transaction")
        with col4:
            st.metric("Stacking Model", "🚨 Fraud" if pred_stacking == 1 else "✅ Normal Transaction")

        st.success("Prediction complete! Both models evaluated your input successfully.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Built using Scikit-learn, XGBoost, and Streamlit | SecurePay Hybrid ML 2025")
