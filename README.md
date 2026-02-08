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
| Logistic Regression | 0.8472 | 0.9023 | 0.7346 | 0.6045 | 0.6633 | 0.5701 |
| Decision Tree | 0.8087 | 0.7478 | 0.6134 | 0.6265 | 0.6199 | 0.4922 |
| KNN | 0.8270 | 0.8595 | 0.6664 | 0.6105 | 0.6372 | 0.5248 |
| Naive Bayes | 0.5826 | 0.8018 | 0.3678 | 0.9414 | 0.5290 | 0.3643 |
| Random Forest | 0.8488 | 0.9007 | 0.7287 | 0.6258 | 0.6734 | 0.5786 |
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
This assignment demonstrates a complete end-to-end machine learning pipeline for binary income classification using the Adult Income dataset. Multiple baseline and ensemble models were trained, evaluated, and compared using comprehensive performance metrics including Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

Results from the notebook experiments show that ensemble-based methods consistently outperform individual classifiers. In particular, **XGBoost achieved the best overall performance**, with the highest Accuracy (0.8633), AUC (0.9227), F1 Score (0.7023), and MCC (0.6180), indicating strong discriminative ability and robust predictive performance.

**Random Forest** also performed strongly, offering a good balance between precision and recall, and outperforming most traditional models. **Logistic Regression** provided a solid baseline with competitive AUC and stable performance, while **KNN** showed moderate results sensitive to feature scaling.

**Naive Bayes**, although achieving very high recall, suffered from low precision, leading to a higher number of false positives, making it less suitable for this use case despite its sensitivity to positive class detection.

Overall, the experimental results confirm that **ensemble models, particularly XGBoost, are the most effective and reliable choice** for predicting income levels on this structured tabular dataset, due to their ability to model complex non-linear relationships and feature interactions.


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
