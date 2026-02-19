"""
EDA script – generates all exploratory visualizations and saves them to reports/figures/.
Run: python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import RAW_CSV, FIGURES_DIR
from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import engineer_features
from src.utils import save_figure, set_plot_style, ensure_dirs


def run_eda():
    ensure_dirs()
    set_plot_style()

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nPlacement distribution:\n{df['Placement_Offer'].value_counts()}")

    # 1. Placement Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Placement_Offer"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Placement Offer")
    ax.set_ylabel("Count")
    ax.set_title("Placement Offer Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    save_figure(fig, "placement_distribution.png")

    # 2. CGPA vs Placement
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Placement_Offer", y="CGPA", palette="Set2", ax=ax)
    ax.set_title("CGPA vs Placement Offer")
    save_figure(fig, "cgpa_vs_placement.png")

    # 3. Internship Impact
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Internships_Count", hue="Placement_Offer", palette="Set1", ax=ax)
    ax.set_title("Internship Count vs Placement Offer")
    ax.legend(title="Placed")
    save_figure(fig, "internship_impact.png")

    # 4. Skills Count Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["total_skills_count"], bins=25, color="#3498db", edgecolor="white")
    ax.set_xlabel("Total Skills Count (normalized)")
    ax.set_ylabel("Frequency")
    ax.set_title("Total Skills Count Distribution")
    save_figure(fig, "skills_count_distribution.png")

    # 5. Certification Impact
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Certifications_Count", hue="Placement_Offer", palette="coolwarm", ax=ax)
    ax.set_title("Certifications Count vs Placement Offer")
    ax.legend(title="Placed")
    save_figure(fig, "certification_impact.png")

    # 6. Salary Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["Salary_Offered_USD"], bins=30, color="#9b59b6", edgecolor="white")
    axes[0].set_title("Salary Distribution")
    axes[0].set_xlabel("Salary (USD)")
    sns.boxplot(y=df["Salary_Offered_USD"], color="#9b59b6", ax=axes[1])
    axes[1].set_title("Salary – Outlier Check")
    fig.tight_layout()
    save_figure(fig, "salary_distribution.png")

    # 7. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    save_figure(fig, "correlation_heatmap.png")

    # 8. Degree Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Degree", hue="Placement_Offer", palette="Set2", ax=ax)
    ax.set_title("Degree vs Placement Offer")
    ax.legend(title="Placed")
    save_figure(fig, "degree_vs_placement.png")

    print("\n✓ All EDA plots saved to reports/figures/")


if __name__ == "__main__":
    run_eda()
