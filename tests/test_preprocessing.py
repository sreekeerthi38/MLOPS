import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from preprocessing import clean_data, build_preprocessing_pipeline, split_features_target, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_raw_df():
    return pd.DataFrame({
        "age": [63, "?", 45], "sex": [1, 0, 1], "cp": [3, 2, 1],
        "trestbps": [145, 130, "?"], "chol": [233, 250, 204],
        "fbs": [1, 0, 0], "restecg": [0, 1, 0], "thalach": [150, 187, 172],
        "exang": [0, 0, 1], "oldpeak": [2.3, 3.5, 1.4], "slope": [0, 0, 2],
        "ca": [0, "?", 0], "thal": [1, 2, 2], "target": [1, 0, 1],
    })


def test_clean_data_handles_missing():
    df = clean_data(make_raw_df())
    assert df["age"].isnull().sum() == 1  # coerced '?' to NaN, not dropped (only target NAs dropped)
    assert len(df) == 3


def test_split_features_target():
    df = clean_data(make_raw_df())
    X, y = split_features_target(df)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(y.unique()).issubset({0, 1})


def test_pipeline_fit_transform():
    df = clean_data(make_raw_df())
    X, y = split_features_target(df)
    pipe = build_preprocessing_pipeline()
    transformed = pipe.fit_transform(X)
    assert transformed.shape[0] == 3


def test_engineered_columns_present_after_clean():
    # Regression guard: training and inference must compute the SAME
    # engineered features. This caught a real API/training mismatch.
    df = clean_data(make_raw_df())
    for col in ("age_group", "chol_bp_ratio", "hr_reserve"):
        assert col in df.columns
