# Child-Privacy-Risk-Detection-for-Applications

An end-to-end machine learning project for identifying digital applications that potentially violate children's privacy regulations (COPPA) using CatBoost with hyperparameter optimization and threshold optimization.

---

## Project Overview

Children increasingly interact with digital applications that may collect or misuse personal information. This project aims to classify digital applications according to their potential risk of violating children's privacy regulations (COPPA).

The proposed solution uses CatBoost, a gradient boosting algorithm that naturally handles categorical features, combined with Optuna for hyperparameter optimization and threshold optimization to improve the detection of high-risk applications in an imbalanced dataset.

---

## Dataset

**Source**

Kaggle — Find IT 2025 Competition

**Target Variable**

- `coppaRisk`
  - `1` = High privacy risk
  - `0` = Low privacy risk

**Dataset Summary**

| Item | Value |
|------|------:|
| Samples | 6,981 |
| Features | 16 |
| Numerical Features | 5 |
| Categorical Features | 11 |
| Class Distribution | ~1 : 9 |

---

## Workflow

<p align="center">
<img src="images/workflow.png" width="850">
</p>

The workflow consists of:

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Train–Validation–Test Split
5. CatBoost Training
6. Hyperparameter Optimization (Optuna)
7. Threshold Optimization
8. Model Evaluation

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate records
- Correcting invalid values
- Handling missing values
- Standardizing categorical variables
- Feature engineering
- Train–Validation–Test split (80:10:10)

---

## Machine Learning Model

The proposed model uses:

- CatBoost Classifier
- Optuna Hyperparameter Optimization
- Threshold Optimization
- Soft Voting Ensemble (Mutually Disjoint Dataset Strategy)

---

## Model Performance

| Metric | Score |
|---------|------:|
| Accuracy | 79% |
| Macro Recall | 0.84 |
| Minority Recall | 0.90 |
| Majority Recall | 0.78 |

The optimized CatBoost model successfully detected **90% of high-risk applications**, significantly improving minority-class detection compared to the baseline model.

---

## Confusion Matrix

<p align="center">
<img src="images/confusion_matrix.png" width="500">
</p>

---

## Feature Importance

<p align="center">
<img src="images/feature_importance.png" width="700">
</p>

The model identifies privacy-related features as the most influential factors in determining whether an application poses a potential privacy risk to children.

---

## Key Insights

- The dataset is highly imbalanced, with approximately a 1:9 ratio between high-risk and low-risk applications.
- Hyperparameter optimization substantially improves model performance.
- Threshold optimization increases recall for the minority class without relying on oversampling techniques.
- CatBoost effectively handles heterogeneous tabular data containing both categorical and numerical features.
- Privacy-related attributes contribute the most to prediction performance.

---

## Recommendations

- Use the model as an initial screening tool for newly published applications.
- Prioritize manual review for applications predicted as high risk.
- Continuously retrain the model using updated application metadata.
- Integrate explainability methods (e.g., SHAP) to improve model transparency.

---

## Repository Structure

```
Child-Privacy-Risk-Detection/
│
├── notebooks/
│   └── Child_Privacy_Risk_Detection.ipynb
│
├── images/
│   ├── workflow.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Optuna
- Matplotlib
- Seaborn

---

## References

- Kaggle Find IT 2025 Competition
- CatBoost Documentation
- Optuna Documentation
- Children's Online Privacy Protection Act (COPPA)

---

## Author

**Devina Sawitri**

Data Science Graduate — Universitas Negeri Surabaya
