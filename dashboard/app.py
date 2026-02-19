"""
Streamlit Interactive Dashboard for Placement & Salary Intelligence.
3-Page dashboard: Placement Prediction, Salary Prediction, EDA Insights.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.config import (
    CLASSIFICATION_MODEL_PATH, REGRESSION_MODEL_PATH,
    FIGURES_DIR, RAW_CSV,
)
from src.utils import load_model
from src.feature_engineering import engineer_features
from src.data_preprocessing import load_raw_data, clean_data


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement & Salary Intelligence",
    page_icon="🎓",
    layout="wide",
)

# ── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def get_clf_model():
    if CLASSIFICATION_MODEL_PATH.exists():
        return load_model(CLASSIFICATION_MODEL_PATH)
    return None


@st.cache_resource
def get_reg_model():
    if REGRESSION_MODEL_PATH.exists():
        return load_model(REGRESSION_MODEL_PATH)
    return None


@st.cache_data
def get_data():
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    return df


def build_feature_df(gender, age, degree, cgpa, internships, projects,
                     certs, tech, comm, apt):
    """Build a DataFrame matching the model's expected features."""
    return pd.DataFrame({
        "Age": [age],
        "CGPA": [cgpa],
        "Internships_Count": [internships],
        "Projects_Count": [projects],
        "Certifications_Count": [certs],
        "Technical_Skills_Score_100": [tech],
        "Communication_Skills_Score_100": [comm],
        "Aptitude_Test_Score_100": [apt],
        "total_skills_count": [(tech + comm + apt) / 100.0],
        "internship_binary": [1 if internships > 0 else 0],
        "certification_count": [certs],
        "academic_index": [cgpa * apt / 100.0],
        "skill_intensity_score": [tech * comm / 100.0],
        "Gender": [gender],
        "Degree": [degree],
    })


def render_input_form(key_prefix=""):
    """Shared input form for student data."""
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key=f"{key_prefix}_gender")
        age = st.slider("Age", 16, 35, 22, key=f"{key_prefix}_age")
        degree = st.selectbox("Degree",
            ["Engineering", "Computer Science", "Business", "Arts", "Data Science"],
            key=f"{key_prefix}_degree")
        cgpa = st.slider("CGPA", 0.0, 4.0, 3.0, 0.01, key=f"{key_prefix}_cgpa")
    with col2:
        internships = st.slider("Internships Count", 0, 5, 1, key=f"{key_prefix}_intern")
        projects = st.slider("Projects Count", 0, 10, 4, key=f"{key_prefix}_proj")
        certs = st.slider("Certifications Count", 0, 5, 2, key=f"{key_prefix}_cert")
    with col3:
        tech = st.slider("Technical Skills (0-100)", 0, 100, 70, key=f"{key_prefix}_tech")
        comm = st.slider("Communication Skills (0-100)", 0, 100, 65, key=f"{key_prefix}_comm")
        apt = st.slider("Aptitude Score (0-100)", 0, 100, 72, key=f"{key_prefix}_apt")

    return gender, age, degree, cgpa, internships, projects, certs, tech, comm, apt


# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", [
    "🎯 Placement Prediction",
    "💰 Salary Prediction",
    "📊 EDA Insights",
])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Placement Prediction
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Placement Prediction":
    st.title("🎯 Placement Prediction")
    st.markdown("Enter student details to predict placement probability.")

    inputs = render_input_form("clf")
    df_input = build_feature_df(*inputs)

    if st.button("Predict Placement", type="primary", key="clf_btn"):
        model = get_clf_model()
        if model is None:
            st.error("Classification model not found. Please train the model first.")
        else:
            prob = model.predict_proba(df_input)[0, 1]
            pred = "✅ Placed" if prob >= 0.5 else "❌ Not Placed"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", pred)
                st.metric("Probability", f"{prob:.1%}")

            with col2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Placement Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ecc71" if prob >= 0.5 else "#e74c3c"},
                        "steps": [
                            {"range": [0, 50], "color": "#fadbd8"},
                            {"range": [50, 100], "color": "#d5f5e3"},
                        ],
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance image
            fi_path = FIGURES_DIR / "feature_importance_clf_gradient_boosting.png"
            if not fi_path.exists():
                fi_path = FIGURES_DIR / "feature_importance_clf_random_forest.png"
            if fi_path.exists():
                st.subheader("Feature Importance")
                st.image(str(fi_path))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Salary Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Salary Prediction":
    st.title("💰 Salary Prediction")
    st.markdown("Predict the expected salary for a placed student.")

    inputs = render_input_form("reg")
    df_input = build_feature_df(*inputs)

    if st.button("Predict Salary", type="primary", key="reg_btn"):
        model = get_reg_model()
        if model is None:
            st.error("Regression model not found. Please train the model first.")
        else:
            salary = model.predict(df_input)[0]

            st.metric("💵 Predicted Salary", f"${salary:,.2f} USD")

            # SHAP plot
            shap_path = FIGURES_DIR / "shap_regression.png"
            if shap_path.exists():
                st.subheader("SHAP Feature Explanation")
                st.image(str(shap_path))

            fi_path = FIGURES_DIR / "feature_importance_reg_gradient_boosting_regressor.png"
            if not fi_path.exists():
                fi_path = FIGURES_DIR / "feature_importance_reg_random_forest_regressor.png"
            if fi_path.exists():
                st.subheader("Feature Importance")
                st.image(str(fi_path))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EDA Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Insights":
    st.title("📊 Exploratory Data Analysis")

    df = get_data()
    st.markdown(f"**Dataset:** {len(df)} students | {len(df.columns)} features")

    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    placed = (df["Placement_Offer"] == "Yes").sum()
    not_placed = (df["Placement_Offer"] == "No").sum()
    col1.metric("Total Students", len(df))
    col2.metric("Placed", placed)
    col3.metric("Not Placed", not_placed)
    col4.metric("Placement Rate", f"{placed / len(df):.1%}")

    st.divider()

    # Show saved EDA figures
    eda_plots = {
        "Placement Distribution": "placement_distribution.png",
        "CGPA vs Placement": "cgpa_vs_placement.png",
        "Internship Impact": "internship_impact.png",
        "Certification Impact": "certification_impact.png",
        "Skills Count Distribution": "skills_count_distribution.png",
        "Salary Distribution": "salary_distribution.png",
        "Degree vs Placement": "degree_vs_placement.png",
        "Correlation Heatmap": "correlation_heatmap.png",
    }

    cols = st.columns(2)
    for i, (title, filename) in enumerate(eda_plots.items()):
        filepath = FIGURES_DIR / filename
        with cols[i % 2]:
            st.subheader(title)
            if filepath.exists():
                st.image(str(filepath))
            else:
                st.info(f"Run EDA first: `python -m src.eda`")
