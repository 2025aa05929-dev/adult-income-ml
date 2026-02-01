import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Adult Income Classification", layout="centered")

st.title("Adult Income Classification – ML Assignment 2")

st.markdown("Upload test CSV data and select a trained model to evaluate predictions.")

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

uploaded_file = st.file_uploader("Upload CSV Test Data", type=["csv"])

model_name = st.selectbox("Select Model", list(models.keys()))

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "income" not in data.columns:
        st.error("CSV must contain 'income' column as target")
    else:
        X = data.drop("income", axis=1)
        y = data["income"]

        model = models[model_name]
        y_pred = model.predict(X)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("Precision:", precision_score(y, y_pred))
        st.write("Recall:", recall_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, y_pred))
