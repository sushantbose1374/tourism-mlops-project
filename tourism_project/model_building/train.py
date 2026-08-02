
"""
=============================================================
Tourism Package Purchase Prediction
Model Training & Experiment Tracking
=============================================================

This module performs the following tasks

1. Load processed datasets
2. Train multiple ML models
3. Hyperparameter optimisation
4. MLflow experiment tracking
5. Best model selection
6. Save deployment artifacts

Author : Sushant Bose
=============================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

ARTIFACT_DIR = "tourism_project/artifacts"

DEPLOYMENT_DIR = "tourism_project/deployment"

TRAIN_PATH = os.path.join(
    ARTIFACT_DIR,
    "train.csv"
)

TEST_PATH = os.path.join(
    ARTIFACT_DIR,
    "test.csv"
)

ENCODER_PATH = os.path.join(
    ARTIFACT_DIR,
    "label_encoders.pkl"
)

MODEL_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "model.pkl"
)

FEATURE_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "feature_columns.pkl"
)

TARGET = "ProdTaken"

os.makedirs(
    DEPLOYMENT_DIR,
    exist_ok=True
)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

print("="*70)
print("Loading Prepared Dataset")
print("="*70)

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

print("Training Shape :", train_df.shape)

print("Testing Shape  :", test_df.shape)

X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]

feature_columns = X_train.columns.tolist()

print("\nNumber of Features :", len(feature_columns))

print("\nTarget Distribution")

print(y_train.value_counts())

# -------------------------------------------------------
# Load Label Encoders
# -------------------------------------------------------

label_encoders = joblib.load(
    ENCODER_PATH
)

print(
    f"\nLoaded {len(label_encoders)} label encoders."
)

# -------------------------------------------------------
# Candidate Models
# -------------------------------------------------------

candidate_models = {

    "Decision Tree":{

        "model":DecisionTreeClassifier(
            random_state=42
        ),

        "params":{

            "max_depth":[3,5,7,None],

            "min_samples_split":[2,5,10]

        }

    },

    "Random Forest":{

        "model":RandomForestClassifier(
            random_state=42
        ),

        "params":{

            "n_estimators":[100,200],

            "max_depth":[5,10,None]

        }

    },

    "AdaBoost":{

        "model":AdaBoostClassifier(
            random_state=42
        ),

        "params":{

            "n_estimators":[50,100],

            "learning_rate":[0.5,1.0]

        }

    },

    "Gradient Boosting":{

        "model":GradientBoostingClassifier(
            random_state=42
        ),

        "params":{

            "n_estimators":[100,200],

            "learning_rate":[0.05,0.1]

        }

    }

}

best_model = None

best_accuracy = 0

best_model_name = ""

results = []

mlflow.set_experiment(
    "Tourism_MLOps_Project"
)
# -------------------------------------------------------
# Model Training
# -------------------------------------------------------

print("\n" + "=" * 70)
print("Model Training & Hyperparameter Optimisation")
print("=" * 70)

for model_name, config in candidate_models.items():

    print(f"\nTraining Model : {model_name}")

    grid_search = GridSearchCV(

        estimator=config["model"],

        param_grid=config["params"],

        cv=5,

        scoring="accuracy",

        n_jobs=-1,

        verbose=0

    )

    grid_search.fit(
        X_train,
        y_train
    )

    trained_model = grid_search.best_estimator_

    predictions = trained_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(f"Best Parameters : {grid_search.best_params_}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    results.append({

        "Model":model_name,

        "Accuracy":accuracy,

        "Precision":precision,

        "Recall":recall,

        "F1 Score":f1,

        "Best Parameters":grid_search.best_params_

    })

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param(
            "Algorithm",
            model_name
        )

        mlflow.log_params(
            grid_search.best_params_
        )

        mlflow.log_metric(
            "Accuracy",
            accuracy
        )

        mlflow.log_metric(
            "Precision",
            precision
        )

        mlflow.log_metric(
            "Recall",
            recall
        )

        mlflow.log_metric(
            "F1 Score",
            f1
        )

        mlflow.sklearn.log_model(
            sk_model=trained_model,
            name="model"
        )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = trained_model

        best_model_name = model_name

print("\n" + "=" * 70)
print("Model Comparison")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print(results_df)

# -------------------------------------------------------
# Save Deployment Bundle
# -------------------------------------------------------

print("\n" + "=" * 70)
print("Saving Deployment Artifacts")
print("=" * 70)

deployment_bundle = {

    "model": best_model,

    "label_encoders": label_encoders,

    "feature_columns": feature_columns,

    "target_column": TARGET,

    "best_model_name": best_model_name,

    "best_accuracy": best_accuracy

}

joblib.dump(
    deployment_bundle,
    MODEL_PATH
)

joblib.dump(
    feature_columns,
    FEATURE_PATH
)

print(f"\nBest Model : {best_model_name}")

print(f"Best Accuracy : {best_accuracy:.4f}")

print(f"\nDeployment bundle saved to:\n{MODEL_PATH}")

print(f"\nFeature list saved to:\n{FEATURE_PATH}")

# -------------------------------------------------------
# Final Model Evaluation
# -------------------------------------------------------

print("\n" + "=" * 70)
print("Final Model Evaluation")
print("=" * 70)

final_predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# -------------------------------------------------------
# Classification Report
# -------------------------------------------------------

print("\n" + "=" * 70)
print("Classification Report")
print("=" * 70)

print(
    classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )
)

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

print("\n" + "=" * 70)
print("Confusion Matrix")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    final_predictions
)

print(cm)

# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n" + "=" * 70)
print("Training Summary")
print("=" * 70)

print(f"Best Model          : {best_model_name}")
print(f"Best Accuracy       : {best_accuracy:.4f}")
print(f"Number of Features  : {len(feature_columns)}")
print(f"Deployment File     : {MODEL_PATH}")

print("\nArtifacts Created")

print(f"✓ Model Bundle      : {MODEL_PATH}")
print(f"✓ Feature Columns   : {FEATURE_PATH}")
print(f"✓ Label Encoders    : {ENCODER_PATH}")

print("\nTraining pipeline completed successfully.")

print("=" * 70)
