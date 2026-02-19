"""
SHAP model interpretability script.
Generates SHAP summary plots for classification and regression models.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import (
    CLASSIFICATION_MODEL_PATH, REGRESSION_MODEL_PATH,
)
from src.data_preprocessing import (
    load_raw_data, clean_data, encode_target,
    get_classification_split, get_regression_split,
)
from src.feature_engineering import engineer_features
from src.utils import load_model, save_figure, ensure_dirs


def _get_feature_names(preprocessor) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out())
        else:
            names.extend(columns)
    return names


def run_shap():
    ensure_dirs()

    # Load data
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_target(df)

    # ── Classification SHAP ───────────────────────────────────────────────────
    print("Computing SHAP values for classification model...")
    clf_model = load_model(CLASSIFICATION_MODEL_PATH)
    _, X_test_clf, _, _ = get_classification_split(df)

    preprocessor_clf = clf_model.named_steps["preprocessor"]
    classifier = clf_model.named_steps["classifier"]
    X_test_transformed = preprocessor_clf.transform(X_test_clf)
    feature_names = _get_feature_names(preprocessor_clf)

    explainer_clf = shap.TreeExplainer(classifier) if hasattr(classifier, "feature_importances_") \
        else shap.LinearExplainer(classifier, X_test_transformed)
    shap_values_clf = explainer_clf.shap_values(X_test_transformed)

    # Handle binary classification SHAP values
    if isinstance(shap_values_clf, list):
        shap_values_clf = shap_values_clf[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_clf, X_test_transformed,
                      feature_names=feature_names, show=False, max_display=10)
    plt.title("SHAP Summary – Classification (Placement)")
    plt.tight_layout()
    save_figure(plt.gcf(), "shap_classification.png")

    # ── Regression SHAP ───────────────────────────────────────────────────────
    print("Computing SHAP values for regression model...")
    reg_model = load_model(REGRESSION_MODEL_PATH)
    _, X_test_reg, _, _ = get_regression_split(df)

    preprocessor_reg = reg_model.named_steps["preprocessor"]
    regressor = reg_model.named_steps["regressor"]
    X_test_reg_transformed = preprocessor_reg.transform(X_test_reg)
    feature_names_reg = _get_feature_names(preprocessor_reg)

    explainer_reg = shap.TreeExplainer(regressor) if hasattr(regressor, "feature_importances_") \
        else shap.LinearExplainer(regressor, X_test_reg_transformed)
    shap_values_reg = explainer_reg.shap_values(X_test_reg_transformed)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_reg, X_test_reg_transformed,
                      feature_names=feature_names_reg, show=False, max_display=10)
    plt.title("SHAP Summary – Regression (Salary)")
    plt.tight_layout()
    save_figure(plt.gcf(), "shap_regression.png")

    print("\n✓ SHAP plots saved to reports/figures/")


if __name__ == "__main__":
    run_shap()
