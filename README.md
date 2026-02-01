# **Adult Income Classification – ML Assignment 2**

---

## **Problem Statement**
The objective of this project is to build and evaluate multiple machine learning classification models to predict whether an individual earns more than **$50K per year** based on demographic and employment-related attributes. The project demonstrates a complete machine learning pipeline including data preprocessing, model training, evaluation, and comparison.

---

## **Dataset Description**
- **Dataset Name:** Adult Income Dataset  
- **Source:** UCI Machine Learning Repository  
- **Raw Dataset Size:** 48,842 instances  
- **Dataset Used in This Project:**  
  - Missing values (`"?"`) removed  
  - **Final Shape:** **30,162 rows × 15 columns**
- **Number of Features:** 14  
- **Target Variable:** `income` (`<=50K`, `>50K`)  
- **Problem Type:** Binary Classification  

The dataset contains demographic and employment-related attributes such as age, education, occupation, hours worked per week, capital gain, and capital loss.

---

## **Data Preprocessing**
The following preprocessing steps were applied:
- Loaded dataset directly from UCI repository
- Replaced `"?"` with missing values and removed incomplete rows
- Label encoded all categorical variables
- Applied **StandardScaler** for feature scaling
- Split data into **80% training and 20% testing** sets using stratified sampling

---

## **Models Implemented**
The following machine learning classification models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble Model)**
6. **XGBoost (Ensemble Model)**

---

## **Evaluation Metrics**
Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## **Model Performance Comparison**

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8175 | 0.8501 | 0.7135 | 0.4461 | 0.5490 | 0.4613 |
| Decision Tree | 0.8067 | 0.7429 | 0.6110 | 0.6158 | 0.6134 | 0.4846 |
| KNN | 0.8190 | 0.8498 | 0.6530 | 0.5826 | 0.6158 | 0.4993 |
| Naive Bayes | 0.7978 | 0.8498 | 0.6986 | 0.3302 | 0.4485 | 0.3798 |
| Random Forest | 0.8538 | 0.9025 | 0.7464 | 0.6252 | 0.6804 | 0.5905 |
| XGBoost | 0.8616 | 0.9204 | 0.7636 | 0.6431 | 0.6982 | 0.6131 |

---

## **Observations**
- **Logistic Regression** performs well overall but has lower recall.
- **Decision Tree** provides balanced precision and recall but lower AUC.
- **KNN** gives stable performance but is sensitive to feature scaling.
- **Naive Bayes** is fast but struggles with recall.
- **Random Forest** shows strong generalization through ensemble learning.
- **XGBoost** achieves the best overall performance across all metrics.

---

## **Conclusion**

This project demonstrates an end-to-end machine learning pipeline for binary income classification, covering data preprocessing, feature scaling, model training, and comprehensive evaluation. The results show that ensemble-based methods, particularly **XGBoost** and **Random Forest**, significantly outperform individual classifiers in terms of accuracy, AUC, and overall predictive robustness.

The comparative analysis highlights the importance of ensemble learning for handling complex, structured tabular data, as these models are better able to capture non-linear relationships and interactions between features. Overall, the findings confirm that advanced ensemble techniques provide superior generalization performance for income prediction tasks on the Adult Income dataset, making them well-suited for real-world classification problems involving heterogeneous demographic and employment attributes.

---

## **Repository Structure**

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
├── Adult_Income_Classification.ipynb
├── README.md
└── requirements.txt

