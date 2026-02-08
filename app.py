import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Adult Income Classification",
    layout="centered"
)

st.title("Adult Income Classification – ML Assignment 2")
st.markdown(
    "Upload test CSV data and select a trained model to evaluate predictions."
)

# --------------------------------------------------
# Load trained models
# --------------------------------------------------
models = {
    "Logistic Regression": joblib.load("models1/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models1/decision_tree.pkl"),
    "KNN": joblib.load("models1/knn.pkl"),
    "Naive Bayes": joblib.load("models1/naive_bayes.pkl"),
    "Random Forest": joblib.load("models1/random_forest.pkl"),
    "XGBoost": joblib.load("models1/xgboost.pkl"),
}

# --------------------------------------------------
# File upload & model selection
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV Test Data",
    type=["csv"]
)

model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

# --------------------------------------------------
# Prediction & evaluation
# --------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "income" not in data.columns:
        st.error("CSV must contain an 'income' column as the target variable.")
    else:
        X = data.drop("income", axis=1)

        # 🔑 label encoding must match training
        y = data["income"].map({
            "<=50K": 0,
            ">50K": 1
        })

        if y.isna().any():
            st.error(
                "Income column contains unexpected labels. "
                "Expected '<=50K' or '>50K'."
            )
        else:
            model = models[model_name]

            # Predictions
            y_pred = model.predict(X)

            # Probabilities (needed for AUC)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                auc_score = roc_auc_score(y, y_prob)
            else:
                auc_score = "Not supported"

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------
            st.subheader("Evaluation Metrics")

            st.write("Accuracy:", accuracy_score(y, y_pred))
            st.write("AUC Score:", auc_score)
            st.write("Precision:", precision_score(y, y_pred))
            st.write("Recall:", recall_score(y, y_pred))
            st.write("F1 Score:", f1_score(y, y_pred))
            st.write("MCC Score:", matthews_corrcoef(y, y_pred))

            # --------------------------------------------------
            # Confusion Matrix
            # --------------------------------------------------
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y, y_pred))
