import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH   = "tourism_project/data/tourism.csv"
TARGET_COL  = "ProdTaken"
DROP_COLS   = ["CustomerID"] # identifier column — no predictive value
TEST_SIZE   = 0.20
RANDOM_SEED = 42


def load_and_clean(path: str) -> pd.DataFrame:

    print(f"\nLoading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Raw shape  : {df.shape}")

    # ── Drop unnecessary columns ─────────────────────────────────────────────
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        print(f"Dropped unnamed index columns: {unnamed_cols}")
    
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)
    print(f"Dropped    : {DROP_COLS}")

    # ── Clean the Gender column ──────────────────────────────────────────────
    # 'Fe Male' is a data-entry error that should be 'Female'.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.strip()
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        print("Gender     : whitespace stripped; 'Fe Male' → 'Female'")

    # ── Impute missing values ────────────────────────────────────────────────
    numeric_cols     = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Exclude the target from imputation
    numeric_features = [c for c in numeric_cols if c != TARGET_COL]

    for col in numeric_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Imputed    : {col!r} with median ({median_val:.2f})")

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Imputed    : {col!r} with mode ({mode_val!r})")

    print(f"Clean shape: {df.shape}")
    print(f"Missing after imputation:\n{df.isnull().sum().to_string()}")
    return df



def split_and_save(df: pd.DataFrame) -> None:

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,          # preserve class balance in each split
    )

    # ── Save splits ──────────────────────────────────────────────────────────
    X_train.to_csv("Xtrain.csv", index=False)
    X_test .to_csv("Xtest.csv",  index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test .to_csv("ytest.csv",  index=False)

    print("\n── Train / Test Splits ──────────────────────────────────────")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")
    print("\n  Target distribution in train split:")
    for lbl, cnt in y_train.value_counts().items():
        print(f"    {lbl} : {cnt} ({100*cnt/len(y_train):.1f} %)")
    print("\n  Target distribution in test split:")
    for lbl, cnt in y_test.value_counts().items():
        print(f"    {lbl} : {cnt} ({100*cnt/len(y_test):.1f} %)")
    print("\n✓  Splits saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    df_clean = load_and_clean(DATA_PATH)
    split_and_save(df_clean)

print("\nprep.py written successfully.")
