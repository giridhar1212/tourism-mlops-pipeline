
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)
import xgboost as xgb

import mlflow
import mlflow.sklearn

# ── Constants ─────────────────────────────────────────────────────────────────
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT    = "tourism_prediction"
MODEL_OUTPUT  = "tourism_project/deployment/model.pkl"
RANDOM_SEED   = 42


# ── Data loading ──────────────────────────────────────────────────────────────
def load_splits():

    X_train = pd.read_csv("Xtrain.csv")
    X_test  = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze()   # Series
    y_test  = pd.read_csv("ytest.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ── Training & evaluation
def train_and_evaluate():


    # ── Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, X_test, y_train, y_test = load_splits()
    preprocessor = build_preprocessor(X_train)

    print(f"\nTraining samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Feature columns  : {X_train.shape[1]}")

    # ── Pipeline definition
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_SEED,
        )),
    ])

    # ── Hyperparameter grid
    param_grid = {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__max_depth":     [3, 5, 7],
        "classifier__learning_rate": [0.05, 0.10, 0.20],
        "classifier__subsample":     [0.7, 0.9],
    }

    print("\nStarting GridSearchCV (this may take a few minutes) ...")

    with mlflow.start_run(run_name="XGBoost_GridSearch") as run:

        # ── Grid search with 5-fold CV, optimised for F1
        grid_search = GridSearchCV(
            estimator  = pipeline,
            param_grid = param_grid,
            cv         = 5,
            scoring    = "f1",
            n_jobs     = -1,
            verbose    = 1,
        )
        grid_search.fit(X_train, y_train)

        best_model  = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_f1  = grid_search.best_score_

        # ── Log all tuned parameters
        print("\nBest hyperparameters found by GridSearchCV:")
        for param, value in best_params.items():
            mlflow.log_param(param, value)
            print(f"  {param}: {value}")

        mlflow.log_param("cv_folds",    5)
        mlflow.log_param("test_size",  0.20)
        mlflow.log_param("random_seed", RANDOM_SEED)

        # ── Evaluate on the held-out test set
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score   (y_test, y_pred)
        f1   = f1_score       (y_test, y_pred)
        auc  = roc_auc_score  (y_test, y_proba)

        # ── Log metrics to MLflow
        mlflow.log_metric("accuracy",      acc)
        mlflow.log_metric("precision",     prec)
        mlflow.log_metric("recall",        rec)
        mlflow.log_metric("f1_score",      f1)
        mlflow.log_metric("roc_auc",       auc)
        mlflow.log_metric("cv_best_f1",    best_cv_f1)

        # ── Log model artefact
        mlflow.sklearn.log_model(best_model, artifact_path="best_xgb_model")

        # ── Console output
        sep = "=" * 60
        print(f"\n{sep}")
        print("  MODEL EVALUATION RESULTS")
        print(sep)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"  CV Best F1: {best_cv_f1:.4f}")

        print(f"\n{'─' * 60}")
        print("  Classification Report")
        print(f"{'─' * 60}")
        print(classification_report(
            y_test, y_pred,
            target_names=["Not Purchased", "Purchased"],
        ))

        print(f"{'─' * 60}")
        print("  Confusion Matrix")
        print(f"{'─' * 60}")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
        print(sep)

        print(f"\nMLflow run ID : {run.info.run_id}")

    # ── Save best model
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT)
    print(f"\n✓  Best model saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    train_and_evaluate()

print("\ntrain.py written successfully.")
