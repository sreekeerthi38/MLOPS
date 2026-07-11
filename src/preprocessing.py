"""Data cleaning + reusable sklearn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from feature_engineering import engineer_features, ENGINEERED_NUMERIC, ENGINEERED_CATEGORICAL

RAW_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
RAW_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL
TARGET = "target"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 'ca' and 'thal' commonly contain '?' in the raw UCI file
    df = df.replace("?", pd.NA)
    for col in RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = engineer_features(df)
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def split_features_target(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    return X, y
