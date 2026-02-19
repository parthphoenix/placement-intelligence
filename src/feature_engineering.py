"""
Feature engineering module.
Creates derived features from the raw dataset.
"""
import pandas as pd
import numpy as np

from src.config import PROCESSED_CSV, PROCESSED_DATA_DIR


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features:
      - total_skills_count: sum of Technical + Communication + Aptitude scores (normalized)
      - internship_binary: 1 if student has any internships, else 0
      - certification_count: alias kept for clarity (same as Certifications_Count)
      - academic_index: CGPA * Aptitude_Test_Score_100 / 100
      - skill_intensity_score: Technical * Communication / 100
    """
    df = df.copy()

    # Total skills count (sum of three score columns, rescaled to 0-3)
    df["total_skills_count"] = (
        df["Technical_Skills_Score_100"]
        + df["Communication_Skills_Score_100"]
        + df["Aptitude_Test_Score_100"]
    ) / 100.0

    # Binary internship flag
    df["internship_binary"] = (df["Internships_Count"] > 0).astype(int)

    # Certification count (keep as-is for consistency)
    df["certification_count"] = df["Certifications_Count"]

    # Academic index: combines GPA with aptitude
    df["academic_index"] = df["CGPA"] * df["Aptitude_Test_Score_100"] / 100.0

    # Skill intensity score: interaction of two skill domains
    df["skill_intensity_score"] = (
        df["Technical_Skills_Score_100"] * df["Communication_Skills_Score_100"]
    ) / 100.0

    return df


def save_processed_data(df: pd.DataFrame):
    """Save the feature-engineered dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Processed data saved → {PROCESSED_CSV}")


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data, clean_data
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    save_processed_data(df)
    print(f"Engineered dataset shape: {df.shape}")
    print(df.head())
