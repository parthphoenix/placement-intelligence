"""
Evaluation utilities for classification and regression models.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score,
)

from src.utils import save_figure, set_plot_style


# ── Classification ────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute classification metrics and return as dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    """Plot and save confusion matrix."""
    set_plot_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed", "Placed"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name}")
    save_figure(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_feature_importance_clf(model, feature_names, model_name: str):
    """Plot feature importance for tree-based classifiers."""
    set_plot_style()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Cannot extract feature importance for {model_name}")
        return

    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        color=sns.color_palette("viridis", len(idx)),
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance – {model_name} (Classification)")
    save_figure(fig, f"feature_importance_clf_{model_name.lower().replace(' ', '_')}.png")


# ── Regression ────────────────────────────────────────────────────────────────

def evaluate_regressor(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute regression metrics and return as dict."""
    y_pred = model.predict(X_test)
    metrics = {
        "Model": model_name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    return metrics


def plot_feature_importance_reg(model, feature_names, model_name: str):
    """Plot feature importance for regression models."""
    set_plot_style()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print(f"Cannot extract feature importance for {model_name}")
        return

    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        color=sns.color_palette("magma", len(idx)),
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance – {model_name} (Regression)")
    save_figure(fig, f"feature_importance_reg_{model_name.lower().replace(' ', '_')}.png")


def print_metrics_table(metrics_list: list[dict]):
    """Pretty-print a table of metrics."""
    df = pd.DataFrame(metrics_list)
    print("\n" + df.to_string(index=False))
    print()
