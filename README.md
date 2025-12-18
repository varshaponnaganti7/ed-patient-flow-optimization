# ed-patient-flow-optimization
Emergency Department patient flow optimization using NLP, ML, and Power BI

# Emergency Department Patient Flow Optimization

This project analyzes Emergency Department (ED) patient flow data to
identify high-acuity patients and provide operational insights using
machine learning and interactive dashboards.

The goal is to support hospital decision-making by understanding:
- When high-acuity patients arrive
- How they arrive
- Which departments handle the highest burden
- Key drivers of patient acuity

---

## 📊 Project Overview

**Dataset**
- Emergency Department visit-level data
- 558,029 patient encounters
- 970+ clinical, demographic, and operational features

**Target Variable**
- `high_acuity`  
  - Defined as ESI level ≤ 2  
  - Binary classification (1 = high acuity, 0 = low acuity)

---

## 🧪 Data Science Workflow

1. Exploratory Data Analysis (EDA)
2. Feature inspection and missingness analysis
3. Target variable creation
4. Preprocessing (categorical encoding, numeric imputation)
5. Baseline model (Logistic Regression)
6. Advanced model (Random Forest)
7. Model evaluation (ROC-AUC, precision, recall)
8. Model interpretability (feature importance, SHAP)
9. Operational dashboard (Tableau)

---

## 🤖 Modeling Results

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | ~0.81 |
| Random Forest | **~0.87** |

Random Forest significantly improved performance and identified key
drivers of high-acuity risk such as:
- Arrival mode (ambulance)
- Department
- Vital signs
- Chief complaints

---

## 📈 Dashboard

An interactive Tableau dashboard summarizes high-acuity patient flow:
- High Acuity by Hour
- High Acuity by Arrival Mode
- High Acuity by Department
- High Acuity by Gender

📍 Dashboard details are documented in `/dashboard/README.md`

---

## 📁 Repository Structure
data/ → Sample dataset and data documentation
notebooks/ → Jupyter notebooks for analysis and modeling
dashboard/ → Dashboard explanation and screenshots

---


---

## 🔒 Data Privacy

The full dataset is not shared due to size and privacy constraints.
A representative sample is provided for reproducibility.

---

## 🧠 Skills Demonstrated

- Python (Pandas, NumPy, scikit-learn)
- Machine Learning (classification, evaluation)
- Feature engineering & preprocessing
- Model interpretability (SHAP, feature importance)
- Data visualization (Tableau)
- Healthcare analytics


