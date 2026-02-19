"""
Classification model training script.
Trains Logistic Regression, Random Forest, and Gradient Boosting for
placement prediction. Uses GridSearchCV with 5-fold CV.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_STATE, CV_FOLDS, CLF_PARAM_GRIDS,
    CLASSIFICATION_MODEL_PATH, CLASSIFICATION_PREPROCESSOR_PATH,
)
from src.data_preprocessing import (
    load_raw_data, clean_data, encode_target,
    get_classification_split, build_preprocessor,
)
from src.feature_engineering import engineer_features
from src.evaluate import (
    evaluate_classifier, plot_confusion_matrix,
    plot_feature_importance_clf, print_metrics_table,
)
from src.utils import save_model, ensure_dirs


MODELS = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}


def train_classification():
    """Full classification training pipeline."""
    ensure_dirs()

    # 1. Load & prepare data
    print("=" * 60)
    print("PLACEMENT CLASSIFICATION TRAINING")
    print("=" * 60)
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_target(df)

    X_train, X_test, y_train, y_test = get_classification_split(df)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 2. Build preprocessor
    preprocessor = build_preprocessor()

    # 3. Train & evaluate each model with GridSearchCV
    results = []
    best_model = None
    best_auc = -1
    best_name = ""

    for name, estimator in MODELS.items():
        print(f"\n{'─' * 40}")
        print(f"Training: {name}")
        print(f"{'─' * 40}")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        # Build param grid with pipeline prefix
        param_grid = {
            f"classifier__{k}": v
            for k, v in CLF_PARAM_GRIDS[name].items()
        }

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=CV_FOLDS,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")

        # Cross-validation score on full training set
        cv_scores = cross_val_score(
            grid.best_estimator_, X_train, y_train,
            cv=CV_FOLDS, scoring="roc_auc",
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Evaluate on test set
        metrics = evaluate_classifier(grid.best_estimator_, X_test, y_test, name)
        metrics["CV_ROC_AUC_mean"] = cv_scores.mean()
        metrics["CV_ROC_AUC_std"] = cv_scores.std()
        results.append(metrics)

        # Track best model
        auc = metrics.get("ROC-AUC", 0)
        if auc and auc > best_auc:
            best_auc = auc
            best_model = grid.best_estimator_
            best_name = name

    # 4. Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Classification)")
    print("=" * 60)
    print_metrics_table(results)

    # 5. Save best model
    print(f"\n✓ Best model: {best_name} (ROC-AUC = {best_auc:.4f})")
    save_model(best_model, CLASSIFICATION_MODEL_PATH)
    save_model(
        best_model.named_steps["preprocessor"],
        CLASSIFICATION_PREPROCESSOR_PATH,
    )

    # 6. Generate plots for best model
    plot_confusion_matrix(best_model, X_test, y_test, best_name)

    # Get feature names from fitted preprocessor
    preprocessor_fitted = best_model.named_steps["preprocessor"]
    feature_names = _get_feature_names(preprocessor_fitted)
    classifier = best_model.named_steps["classifier"]
    plot_feature_importance_clf(classifier, feature_names, best_name)

    print("\n✓ Classification training complete!")
    return best_model, results


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


if __name__ == "__main__":
    train_classification()
