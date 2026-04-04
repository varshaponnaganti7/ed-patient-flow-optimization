#  Emergency Department High-Acuity Patient Flow Optimization

##  Overview

This project focuses on predicting **high-acuity patients** in emergency departments using machine learning and analyzing patient flow using an interactive dashboard.

The goal is to support **clinical decision-making and resource optimization** by identifying critical patients early and understanding operational patterns.

---

##  Objectives

* Predict whether a patient is **high-acuity (ESI ≤ 2)**
* Reduce missed critical cases (false negatives)
* Provide **data-driven insights** into emergency department flow
* Combine **machine learning + visualization (Tableau)**

---

##  Dashboard

An interactive Tableau dashboard is available:

🔗 https://public.tableau.com/app/profile/varsha.ponnaganti6764/viz/emergency_17726665699380/Dashboard1

### Key Insights:

* High-acuity patient trends by hour
* Arrival mode impact (ambulance vs walk-in)
* Department-level workload distribution
* Demographic patterns

---

##  Machine Learning Approach

### Target Variable

* `high_acuity` (1 if ESI ≤ 2, else 0)

### Features Used

* Demographics (age, gender, race, etc.)
* Arrival information (mode, time)
* Chief complaint indicators (cc_* features)

### Models

* Logistic Regression (baseline)
* Random Forest (final model)

### Performance (Random Forest)

* ROC-AUC: ~0.86
* Recall (high-acuity): ~0.80+
* Focus: **maximize recall to avoid missing critical patients**

---

##  Project Structure

```plaintext
ed-patient-flow-optimization/
│
├── data/
│   ├── sample_ed_data.csv
│   └── random_test_input.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── 00_r_to_csv_conversion.ipynb
│   └── 01_model_training.ipynb
│
├── src/
│   └── predict.py
│
├── dashboard/
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## 📁 Dataset

Original dataset is provided in R format:

🔗 https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data

### Notes:

* Full dataset is **not included** due to size constraints
* A sample dataset is provided for testing

---

##  How to Run

### 1. (Optional) Convert R dataset to CSV

Run:

```bash
notebooks/00_r_to_csv_conversion.ipynb
```

---

### 2. Train the model

Run:

```bash
notebooks/01_model_training.ipynb
```

This will generate:

```plaintext
models/rf_pipeline.pkl
```

---

### 3. Run predictions

```bash
python src/predict.py
```

Enter input file path when prompted:

```plaintext
data/sample_ed_data.csv
```

Output:

```plaintext
predictions.csv
```

---

##  Output

The prediction output includes:

* `predicted_high_acuity` (0 or 1)
* `probability_high_acuity`
* `risk_level` (Low / Medium / High)

---

##  Key Design Decisions

* **Handled class imbalance** using class weighting
* **Optimized for recall** (critical in healthcare)
* **Prevented data leakage** by removing ESI features
* Built a **robust pipeline** to handle missing/extra columns
* Enabled predictions on **partial datasets**

---

##  Notes

* Model file is not included due to size limits
* Users must run training notebook to generate the model
* Sample dataset provided for quick testing

---

##  Future Improvements

* Deploy as a web application (Streamlit / FastAPI)
* Real-time prediction integration
* Advanced explainability (SHAP)
* Time-series forecasting for ED load

---

##  Summary

This project demonstrates an **end-to-end machine learning workflow**:

* Data preprocessing
* Model development
* Evaluation
* Deployment-ready prediction pipeline
* Business dashboard integration

---

##  Author

Varsha Ponnaganti
