]import gradio as gr
import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")

feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"
]

def predict(*features):
    data = np.array(features).reshape(1, -1)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        return f"🚨 Fraud Transaction\nFraud Probability: {prob:.2%}"
    else:
        return f"✅ Genuine Transaction\nFraud Probability: {prob:.2%}"

inputs = [
    gr.Number(label=name, value=0)
    for name in feature_names
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="💳 Credit Card Fraud Detection",
    description="XGBoost-based Credit Card Fraud Detection System"
)

demo.launch()