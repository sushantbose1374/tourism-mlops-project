
"""
Data Preparation Script

This script prepares the tourism dataset for model training by
performing data cleaning, feature engineering and train-test splitting.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_PATH = "tourism_project/data/tourism.csv"

ARTIFACT_PATH = "tourism_project/artifacts"

TARGET = "ProdTaken"

# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Original Dataset Shape : {df.shape}")

# ------------------------------------------------------------------
# Remove unnecessary columns
# ------------------------------------------------------------------

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"Shape after removing redundant columns : {df.shape}")

# ------------------------------------------------------------------
# Missing Value Treatment
# ------------------------------------------------------------------

print("\nHandling missing values...")

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in numerical_columns:
    df[column].fillna(df[column].median(), inplace=True)

for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

print("Missing values handled successfully.")

# ------------------------------------------------------------------
# Encode categorical features
# ------------------------------------------------------------------

print("\nEncoding categorical variables...")

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

print(f"Encoded {len(categorical_columns)} categorical columns.")

# Save encoders for deployment

os.makedirs(ARTIFACT_PATH, exist_ok=True)

joblib.dump(
    encoders,
    os.path.join(ARTIFACT_PATH, "label_encoders.pkl")
)

# ------------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------------

X = df.drop(columns=[TARGET])

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)

test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(
    os.path.join(ARTIFACT_PATH, "train.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(ARTIFACT_PATH, "test.csv"),
    index=False
)

print("\nTrain Dataset :", train_df.shape)

print("Test Dataset  :", test_df.shape)

print("\nData preparation completed successfully.")
