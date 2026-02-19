"""
FastAPI backend for Placement & Salary Intelligence.
Provides POST endpoints for placement and salary prediction.
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import StudentInput, PlacementResponse, SalaryResponse
from src.config import CLASSIFICATION_MODEL_PATH, REGRESSION_MODEL_PATH
from src.utils import load_model

app = FastAPI(
    title="Placement & Salary Intelligence API",
    description="Predict student placement status and expected salary.",
    version="1.0.0",
)

# ── Load models at startup ────────────────────────────────────────────────────
clf_model = None
reg_model = None


@app.on_event("startup")
def load_models():
    global clf_model, reg_model
    if CLASSIFICATION_MODEL_PATH.exists():
        clf_model = load_model(CLASSIFICATION_MODEL_PATH)
        print(f"✓ Classification model loaded from {CLASSIFICATION_MODEL_PATH}")
    else:
        print(f"⚠ Classification model not found at {CLASSIFICATION_MODEL_PATH}")
    if REGRESSION_MODEL_PATH.exists():
        reg_model = load_model(REGRESSION_MODEL_PATH)
        print(f"✓ Regression model loaded from {REGRESSION_MODEL_PATH}")
    else:
        print(f"⚠ Regression model not found at {REGRESSION_MODEL_PATH}")


def _build_features(student: StudentInput) -> pd.DataFrame:
    """Convert API input to the feature DataFrame expected by the model."""
    data = {
        "Age": [student.Age],
        "CGPA": [student.CGPA],
        "Internships_Count": [student.Internships_Count],
        "Projects_Count": [student.Projects_Count],
        "Certifications_Count": [student.Certifications_Count],
        "Technical_Skills_Score_100": [student.Technical_Skills_Score_100],
        "Communication_Skills_Score_100": [student.Communication_Skills_Score_100],
        "Aptitude_Test_Score_100": [student.Aptitude_Test_Score_100],
        # Engineered features
        "total_skills_count": [
            (student.Technical_Skills_Score_100
             + student.Communication_Skills_Score_100
             + student.Aptitude_Test_Score_100) / 100.0
        ],
        "internship_binary": [1 if student.Internships_Count > 0 else 0],
        "certification_count": [student.Certifications_Count],
        "academic_index": [student.CGPA * student.Aptitude_Test_Score_100 / 100.0],
        "skill_intensity_score": [
            student.Technical_Skills_Score_100 * student.Communication_Skills_Score_100 / 100.0
        ],
        "Gender": [student.Gender],
        "Degree": [student.Degree],
    }
    return pd.DataFrame(data)


@app.get("/")
def health():
    return {"status": "ok", "models_loaded": {
        "classification": clf_model is not None,
        "regression": reg_model is not None,
    }}


@app.post("/predict_placement", response_model=PlacementResponse)
def predict_placement(student: StudentInput):
    if clf_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    df = _build_features(student)
    prob = clf_model.predict_proba(df)[0, 1]
    pred = "Placed" if prob >= 0.5 else "Not Placed"

    return PlacementResponse(
        placement_prediction=pred,
        placement_probability=round(float(prob), 4),
    )


@app.post("/predict_salary", response_model=SalaryResponse)
def predict_salary(student: StudentInput):
    if reg_model is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded")

    df = _build_features(student)
    salary = reg_model.predict(df)[0]

    return SalaryResponse(
        predicted_salary=round(float(salary), 2),
    )
