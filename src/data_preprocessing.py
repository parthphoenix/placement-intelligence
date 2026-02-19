"""
Data preprocessing module.
Handles loading, cleaning, splitting, and building sklearn pipelines.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from src.config import (
    RAW_CSV, RANDOM_STATE, TEST_SIZE,
    TARGET_CLASSIFICATION, TARGET_REGRESSION,
    ALL_NUMERIC_FEATURES, ALL_CATEGORICAL_FEATURES, DROP_COLUMNS,
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop ID column, handle missing values."""
    df = df.copy()
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    # Drop rows with any missing target values
    df.dropna(subset=[TARGET_CLASSIFICATION, TARGET_REGRESSION], inplace=True)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Placement_Offer as binary (Yes=1, No=0)."""
    df = df.copy()
    df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].map({"Yes": 1, "No": 0})
    return df


def get_classification_split(df: pd.DataFrame):
    """
    Stratified 80/20 split for classification.
    Returns X_train, X_test, y_train, y_test.
    """
    feature_cols = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_CLASSIFICATION]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_regression_split(df: pd.DataFrame):
    """
    Split for regression — use only placed students.
    Returns X_train, X_test, y_train, y_test.
    """
    placed = df[df[TARGET_CLASSIFICATION] == 1].copy()
    feature_cols = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
    X = placed[feature_cols]
    y = placed[TARGET_REGRESSION]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_preprocessor():
    """
    Build a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, ALL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor
