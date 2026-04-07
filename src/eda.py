"""Exploratory data analysis helpers for the laptop price predictor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR, TARGET_COLUMN
from src.preprocess import run_pipeline

sns.set_theme(style="whitegrid")


def _save_current_plot(filename: str) -> None:
    """Save the active matplotlib figure to the reports directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(REPORTS_DIR) / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Plot raw and log-transformed price distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, ax=axes[0], color="crimson")
    axes[0].set_title("Price Distribution")
    sns.histplot(np.log(df[TARGET_COLUMN]), kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Log Price Distribution")
    figure.suptitle("Laptop Price Distribution Overview")
    _save_current_plot("price_distribution.png")


def plot_categorical_counts(df: pd.DataFrame, col: str) -> None:
    """Create and save a count plot for a categorical feature."""
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x=col, palette="viridis", order=df[col].value_counts().index)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Count Plot for {col}")
    _save_current_plot(f"count_{col.lower().replace(' ', '_')}.png")


def plot_price_by_feature(df: pd.DataFrame, col: str) -> None:
    """Create and save a bar plot of average price by category."""
    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=df,
        x=col,
        y=TARGET_COLUMN,
        palette="magma",
        estimator=np.mean,
        errorbar=None,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Average Price by {col}")
    _save_current_plot(f"avg_price_by_{col.lower().replace(' ', '_')}.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create and save a heatmap of numeric feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="plasma", fmt=".2f")
    plt.title("Correlation Heatmap")
    _save_current_plot("correlation_heatmap.png")


def run_eda(df: pd.DataFrame) -> None:
    """Run a standard EDA workflow and save plots to the reports directory."""
    plot_price_distribution(df)
    plot_categorical_counts(df, "Company")
    plot_categorical_counts(df, "TypeName")
    engineered = run_pipeline(df)
    plot_price_by_feature(engineered, "CPU_name")
    plot_price_by_feature(engineered, "Gpu brand")
    plot_correlation_heatmap(engineered)
