import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Adult Income Prediction", layout="centered")

st.title("Adult Income Classification")
st.write("Machine Learning Models on UCI Adult Dataset")

# Load model
model = joblib.load("models/random_forest.pkl")

st.subheader("Model Used")
st.write("Random Forest Classifier")

st.subheader("Evaluation Metrics")
metrics = {
    "Accuracy": 0.86,
    "AUC": 0.91,
    "Precision": 0.74,
    "Recall": 0.62,
    "F1 Score": 0.67,
    "MCC": 0.58
}

st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Score"]))

st.success("Model loaded successfully 🎉")
