#!/usr/bin/env python
# coding: utf-8

# # R to CSV Conversion
# 
# This notebook converts the original emergency department dataset from R format (.rdata) into CSV format for downstream machine learning tasks.
# 
# Dataset source:
# https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data

# In[2]:


import pyreadr
import pandas as pd
import os

# CONFIG s)

rdata_path = "../data/5v_cleandf.rdata"
output_csv_path = "../data/ed_patient_flow.csv"

# CHECK FILE EXISTS

if not os.path.exists(rdata_path):
    raise FileNotFoundError(
        "Dataset not found. Please download from Kaggle and place it in the data/ folder."
    )

# LOAD R DATA

result = pyreadr.read_r(rdata_path)

print("Keys in R file:", result.keys())

df = result["df"]

print("Data loaded successfully.")
print("Shape:", df.shape)

# MISSING VALUE CHECK

missing_counts = df.isna().sum().sort_values(ascending=False)

high_missing = (missing_counts > 0.99 * len(df)).sum()
print("Columns with >99% missing values:", high_missing)

# SAVE CSV

df.to_csv(output_csv_path, index=False)

print(f"CSV saved successfully at: {output_csv_path}")

# VERIFY SAVE

df_check = pd.read_csv(output_csv_path)
print("Reloaded CSV shape:", df_check.shape)

