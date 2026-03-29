#!/usr/bin/env python
# coding: utf-8

# # ED Patient Flow Prediction (Machine Learning)
# 
# This notebook builds a machine learning model to predict high-acuity emergency department patients using clinical and demographic data.
# 
# - Dataset: Hospital Triage and Patient History (Kaggle)
# - Target: High Acuity (ESI ≤ 2)

# In[2]:


import numpy as np
import pandas as pd
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = "42"


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import joblib


# In[4]:


if os.path.exists("../data/ed_patient_flow.csv"):
    df = pd.read_csv("../data/ed_patient_flow.csv")
    print("Loaded FULL dataset")
else:
    df = pd.read_csv("../data/sample_ed_data.csv")
    print("Loaded SAMPLE dataset")

print("Dataset shape:", df.shape)


# In[5]:


df["esi_num"] = pd.to_numeric(df["esi"], errors="coerce")
df = df.dropna(subset=["esi_num"])

df["high_acuity"] = (df["esi_num"] <= 2).astype(int)

print(df["high_acuity"].value_counts(normalize=True))


# In[6]:


df = df.drop(columns=["esi", "esi_num"])


# In[7]:


X = df.drop(columns=["high_acuity"])
y = df["high_acuity"]

X = X.copy()
X["age"] = pd.to_numeric(X["age"], errors="coerce")


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


# In[9]:


categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# In[10]:


log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED
    ))
])

log_reg.fit(X_train, y_train)


# In[11]:


y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("Logistic Regression Results:\n")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[12]:


threshold = 0.40
y_pred_adj = (y_proba >= threshold).astype(int)

print("Threshold = 0.40 Results:\n")
print(confusion_matrix(y_test, y_pred_adj))
print(classification_report(y_test, y_pred_adj))


# In[13]:


rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    ))
])

rf.fit(X_train, y_train)


# In[14]:


y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Results:\n")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))


# In[15]:


model = rf.named_steps["model"]
feature_names = rf.named_steps["preprocessor"].get_feature_names_out()

importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values(ascending=False).head(15)


# In[16]:


import os
import joblib

model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "rf_pipeline.pkl")

joblib.dump(rf, model_path)

print(f"Model saved at: {model_path}")


# In[ ]:




