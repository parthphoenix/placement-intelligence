"""
Regression model training script.
Trains Linear Regression, Ridge, Random Forest Regressor,
and Gradient Boosting Regressor for salary prediction.
Uses only placed students. GridSearchCV with 5-fold CV.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_STATE, CV_FOLDS, REG_PARAM_GRIDS,
    REGRESSION_MODEL_PATH, REGRESSION_PREPROCESSOR_PATH,
)
from src.data_preprocessing import (
    load_raw_data, clean_data, encode_target,
    get_regression_split, build_preprocessor,
)
from src.feature_engineering import engineer_features
from src.evaluate import (
    evaluate_regressor, plot_feature_importance_reg,
    print_metrics_table,
)
from src.utils import save_model, ensure_dirs


MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def train_regression():
    """Full regression training pipeline."""
    ensure_dirs()

    print("=" * 60)
    print("SALARY REGRESSION TRAINING")
    print("=" * 60)
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_target(df)

    X_train, X_test, y_train, y_test = get_regression_split(df)
    print(f"Placed students → Train: {len(X_train)} | Test: {len(X_test)}")

    preprocessor = build_preprocessor()

    results = []
    best_model = None
    best_r2 = -np.inf
    best_name = ""

    for name, estimator in MODELS.items():
        print(f"\n{'─' * 40}")
        print(f"Training: {name}")
        print(f"{'─' * 40}")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ])

        param_grid = {
            f"regressor__{k}": v
            for k, v in REG_PARAM_GRIDS[name].items()
        }

        if param_grid:
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=CV_FOLDS,
                scoring="r2",
                n_jobs=-1,
                refit=True,
            )
        else:
            # Linear Regression has no hyperparams to tune
            grid = GridSearchCV(
                pipe,
                param_grid={},
                cv=CV_FOLDS,
                scoring="r2",
                n_jobs=-1,
                refit=True,
            )

        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV R²: {grid.best_score_:.4f}")

        cv_scores = cross_val_score(
            grid.best_estimator_, X_train, y_train,
            cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
        )
        print(f"  CV RMSE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

        metrics = evaluate_regressor(grid.best_estimator_, X_test, y_test, name)
        metrics["CV_RMSE_mean"] = -cv_scores.mean()
        metrics["CV_RMSE_std"] = cv_scores.std()
        results.append(metrics)

        r2 = metrics["R2"]
        if r2 > best_r2:
            best_r2 = r2
            best_model = grid.best_estimator_
            best_name = name

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Regression)")
    print("=" * 60)
    print_metrics_table(results)

    print(f"\n✓ Best model: {best_name} (R² = {best_r2:.4f})")
    save_model(best_model, REGRESSION_MODEL_PATH)
    save_model(
        best_model.named_steps["preprocessor"],
        REGRESSION_PREPROCESSOR_PATH,
    )

    preprocessor_fitted = best_model.named_steps["preprocessor"]
    feature_names = _get_feature_names(preprocessor_fitted)
    regressor = best_model.named_steps["regressor"]
    plot_feature_importance_reg(regressor, feature_names, best_name)

    print("\n✓ Regression training complete!")
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
    train_regression()
