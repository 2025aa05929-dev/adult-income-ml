# Adult Income Classification – ML Assignment 2

---

## Problem Statement
The objective of this assignment is to build and evaluate multiple machine learning classification models to predict whether an individual earns more than **$50K per year** based on demographic and employment-related attributes. This project demonstrates a complete end-to-end ML workflow including data preprocessing, model training, evaluation, and comparison.  

---

## Dataset Description
- **Dataset Name:** Adult Income Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)  
- **Raw Dataset Size:** 48,842 instances  
- **Final Dataset Used:**  
  - Rows with missing values removed  
  - **Final Shape:** 30,162 rows × 15 columns  
- **Number of Features:** 14  
- **Target Variable:** `income` (`<=50K`, `>50K`)  
- **Problem Type:** Binary Classification  

The dataset contains demographic and employment-related attributes such as age, education, occupation, hours worked per week, capital gain, and capital loss.  

---

## Models Used
The following six machine learning classification models were implemented:

1. **Logistic Regression**  
2. **Decision Tree Classifier**  
3. **K-Nearest Neighbors (KNN)**  
4. **Naive Bayes (Gaussian)**  
5. **Random Forest (Ensemble Model)**  
6. **XGBoost (Ensemble Model)**  

Each model was evaluated using **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.  

---

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8475 | 0.9022 | 0.7354 | 0.6052 | 0.6640 | 0.5711 |
| Decision Tree | 0.8134 | 0.7536 | 0.6229 | 0.6345 | 0.6286 | 0.5040 |
| KNN | 0.8270 | 0.8595 | 0.6664 | 0.6105 | 0.6372 | 0.5248 |
| Naive Bayes | 0.5826 | 0.8018 | 0.3678 | 0.9414 | 0.5290 | 0.3643 |
| Random Forest | 0.8495 | 0.9004 | 0.7299 | 0.6278 | 0.6750 | 0.5806 |
| XGBoost | 0.8633 | 0.9227 | 0.7667 | 0.6478 | 0.7023 | 0.6180 |

---

## Observations on Model Performance

| ML Model | Observation |
|-----------|------------|
| Logistic Regression | Performs well overall with balanced metrics, but recall is slightly lower than precision. |
| Decision Tree | Provides balanced precision and recall but lower AUC compared to ensemble models. |
| KNN | Stable performance; sensitive to feature scaling. |
| Naive Bayes | Very high recall but low precision; prone to false positives. |
| Random Forest | Strong ensemble performance; good balance of precision, recall, and overall metrics. |
| XGBoost | Best overall performance across all metrics; most robust and recommended model. |

---

## Conclusion
This assignment demonstrates an end-to-end machine learning pipeline for binary income classification. Results show that **ensemble methods**, especially **XGBoost** and **Random Forest**, outperform individual classifiers in terms of accuracy, AUC, and overall predictive robustness.  

Ensemble models are able to capture complex non-linear relationships and interactions between features, making them ideal for structured tabular data like the Adult Income dataset. **XGBoost** emerged as the most accurate and reliable model for predicting income above $50K.

---

## Repository Structure

```text
adult-income-ml/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── streamlit_app.py
├── requirements.txt
└── README.md
