#!/usr/bin/env python
# coding: utf-8

# test dataset

# In[13]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import os

# =========================
# CONFIG
# =========================

MODEL_PATH = "/Users/varshaponnaganti/Desktop/projects/healthcare/P_HC_1.2_Emergency_EDwait/ed-patient-flow-ml/models/rf_pipeline.pkl"

# =========================
# LOAD MODEL
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train the model first.")

model = joblib.load(MODEL_PATH)

# Get expected columns from pipeline
expected_cols = model.named_steps["preprocessor"].feature_names_in_

# =========================
# VALIDATION
# =========================

def validate_input(df):
    if df.empty:
        raise ValueError("Input dataset is empty")
    
    if df.shape[1] < 5:
        raise ValueError("Dataset has too few columns")
    
    return df

# =========================
# CLEANING
# =========================

def clean_data(df):
    df = df.copy()
    
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    
    return df

# =========================
# ALIGN DATA (KEY PART 🔥)
# =========================

def align_columns(df, expected_cols):
    df = df.copy()
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_cols]
    
    # Create missing columns at once (FAST)
    missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
    
    # Combine in one step
    df = pd.concat([df, missing_df], axis=1)
    
    # Keep only expected columns
    df = df[expected_cols]
    
    print(f"Added {len(missing_cols)} missing columns")
    print(f"Ignored {len(extra_cols)} extra columns")
    
    return df

# =========================
# RISK CATEGORIZATION
# =========================

def categorize_risk(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

# =========================
# MAIN FUNCTION
# =========================

def predict(file_path, output_path="predictions.csv"):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    df = validate_input(df)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Aligning columns...")
    X = align_columns(df, expected_cols)
    
    print("Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    df["predicted_high_acuity"] = predictions
    df["probability_high_acuity"] = probabilities
    df["risk_level"] = [categorize_risk(p) for p in probabilities]
    
    df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    
    input_file = input("Enter input CSV file path: ")
    predict(input_file)


# In[ ]:




