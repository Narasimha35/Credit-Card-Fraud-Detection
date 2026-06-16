import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Predict whether a transaction is Fraudulent or Genuine using XGBoost.")

st.sidebar.header("Transaction Features")

features = []

feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"
]

for feature in feature_names:
    value = st.sidebar.number_input(
        feature,
        value=0.0,
        format="%.4f"
    )
    features.append(value)

if st.button("Predict Transaction"):

    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"🚨 Fraudulent Transaction\n\nFraud Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Genuine Transaction\n\nFraud Probability: {probability:.2%}"
        )

    st.metric(
        label="Fraud Probability",
        value=f"{probability:.2%}"
    )