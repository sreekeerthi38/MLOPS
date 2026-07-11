"""
Engineered features beyond the raw 13 UCI columns.

Rationale (put this in your report's Feature Engineering section):
- age_group: clinical risk bands (young/middle/senior/elderly) capture
  non-linear age risk that a raw scaler can't, and give tree models a
  ready-made split point.
- chol_bp_ratio: cholesterol relative to resting BP — a simple proxy for
  combined cardiovascular strain rather than treating the two as independent.
- hr_reserve: 220 - age - thalach, an approximation of how far a patient's
  achieved max heart rate falls short of age-predicted max. Low/negative
  reserve is a recognized informal cardiology heuristic.
"""
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 40, 55, 65, 120],
        labels=["young", "middle", "senior", "elderly"],
    )

    df["chol_bp_ratio"] = df["chol"] / df["trestbps"].replace(0, pd.NA)

    df["hr_reserve"] = 220 - df["age"] - df["thalach"]

    return df


ENGINEERED_NUMERIC = ["chol_bp_ratio", "hr_reserve"]
ENGINEERED_CATEGORICAL = ["age_group"]
