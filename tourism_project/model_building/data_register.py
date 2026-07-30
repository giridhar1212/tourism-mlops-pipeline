import sys
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


# ── Validation & Summary ──────────────────────────────────────────────────────
def validate_and_register(path: str) -> None:
    """Validate dataset columns and print a structured summary."""

    print(f"\nLoading dataset from: {path}")
    df = pd.read_csv(path)

    # ── Column validation ────────────────────────────────────────────────────
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"\n[ERROR] Missing expected columns: {missing_cols}")
        sys.exit(1)                          # fail the CI job if columns differ
    print("✓  All expected columns are present.")

    # ── Dataset summary ──────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DATASET SUMMARY")
    print(sep)
    print(f"  File path  : {path}")
    print(f"  Rows       : {df.shape[0]}")
    print(f"  Columns    : {df.shape[1]}")

    print(f"\n{'─' * 60}")
    print("  Column Data Types")
    print(f"{'─' * 60}")
    print(df.dtypes.to_string())

    print(f"\n{'─' * 60}")
    print("  Missing Values per Column")
    print(f"{'─' * 60}")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  No missing values found.")

    print(f"\n{'─' * 60}")
    print("  Target Column Distribution  (ProdTaken)")
    print(f"{'─' * 60}")
    vc = df["ProdTaken"].value_counts()
    for label, count in vc.items():
        pct = 100 * count / len(df)
        print(f"  {label} ({['Not Purchased', 'Purchased'][label]}) : {count:>5}  ({pct:.1f} %)")

    print(f"\n{'─' * 60}")
    print("  Numeric Feature Statistics")
    print(f"{'─' * 60}")
    print(df.describe().round(2).to_string())

    print(f"\n{sep}")
    print("  ✓  Dataset registration completed successfully.")
    print(sep)


if __name__ == "__main__":
    validate_and_register(DATA_PATH)

print("data_register.py written successfully.")
