"""
Configuration module for the Placement Intelligence project.
Centralizes all paths, constants, and hyperparameter grids.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CSV = RAW_DATA_DIR / "Student_Placement_Skills_2025.csv"
PROCESSED_CSV = PROCESSED_DATA_DIR / "processed_data.csv"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.pkl"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.pkl"
CLASSIFICATION_PREPROCESSOR_PATH = MODELS_DIR / "classification_preprocessor.pkl"
REGRESSION_PREPROCESSOR_PATH = MODELS_DIR / "regression_preprocessor.pkl"

# ── Report paths ──────────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── Column definitions ───────────────────────────────────────────────────────
TARGET_CLASSIFICATION = "Placement_Offer"
TARGET_REGRESSION = "Salary_Offered_USD"

NUMERIC_FEATURES = [
    "Age", "CGPA", "Internships_Count", "Projects_Count",
    "Certifications_Count", "Technical_Skills_Score_100",
    "Communication_Skills_Score_100", "Aptitude_Test_Score_100",
]

CATEGORICAL_FEATURES = ["Gender", "Degree"]

ENGINEERED_FEATURES = [
    "total_skills_count",
    "internship_binary",
    "certification_count",
    "academic_index",
    "skill_intensity_score",
]

# All features used in modeling (numeric originals + engineered + categorical)
ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "total_skills_count", "internship_binary",
    "certification_count", "academic_index", "skill_intensity_score",
]
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES

DROP_COLUMNS = ["Student_ID"]

# ── Hyperparameter grids ─────────────────────────────────────────────────────
CLF_PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
}

REG_PARAM_GRIDS = {
    "Linear Regression": {},  # no hyperparams to tune
    "Ridge": {
        "alpha": [0.1, 1, 10, 100],
    },
    "Random Forest Regressor": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
}
