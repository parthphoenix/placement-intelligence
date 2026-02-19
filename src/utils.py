"""
Utility functions for the Placement Intelligence project.
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import FIGURES_DIR, MODELS_DIR


def ensure_dirs():
    """Create all required directories if they don't exist."""
    from src.config import (
        PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
    )
    for d in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_model(model, path):
    """Persist a model to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(path):
    """Load a model from disk."""
    return joblib.load(path)


def save_figure(fig, name: str, dpi: int = 150):
    """Save a matplotlib figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / name
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {filepath}")


def set_plot_style():
    """Apply a consistent plotting style."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })
