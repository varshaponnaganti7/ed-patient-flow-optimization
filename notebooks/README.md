# Notebooks

This folder contains Jupyter notebooks used for data exploration,
feature engineering, modeling, and evaluation.

## Notebook Index

### `01_ed_data_loading_and_exploration.ipynb`
- Loads Emergency Department patient flow data
- Performs initial exploratory data analysis (EDA)
- Examines missing values and data types
- Creates the `high_acuity` target variable (ESI ≤ 2)
- Basic feature inspection and cleaning

### `03_modeling_and_evaluation.ipynb`
- Feature preprocessing (encoding and imputation)
- Train-test split
- Baseline Logistic Regression model
- Random Forest model
- Model evaluation:
  - Confusion matrix
  - Precision, recall, F1-score
  - ROC-AUC
- Feature importance and SHAP-based interpretation
