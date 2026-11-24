"""Matplotlib and Seaborn figures used in exploratory analysis.

Functions return a Matplotlib figure. The caller can display or save it.
Nothing is drawn with a blocking ``show`` call.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _figure():
    return plt.subplots(figsize=(8, 5))


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    fig, ax = _figure()
    ax.hist(df[column].dropna(), bins=bins, edgecolor="white")
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig, ax


def plot_boxplot(df: pd.DataFrame, column: str):
    fig, ax = _figure()
    ax.boxplot(df[column].dropna(), orientation="vertical")
    ax.set_title(f"Boxplot of {column}")
    fig.tight_layout()
    return fig, ax


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None):
    fig, ax = _figure()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"{y} against {x}")
    fig.tight_layout()
    return fig, ax


def plot_count(df: pd.DataFrame, column: str):
    fig, ax = _figure()
    sns.countplot(data=df, x=column, ax=ax)
    ax.set_title(f"Counts of {column}")
    fig.tight_layout()
    return fig, ax


def plot_distribution(df: pd.DataFrame, column: str):
    fig, ax = _figure()
    sns.histplot(data=df, x=column, kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    fig.tight_layout()
    return fig, ax


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson"):
    fig, ax = _figure()
    corr = df.select_dtypes(include="number").corr(method=method)
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=ax)
    ax.set_title(f"{method.title()} correlation")
    fig.tight_layout()
    return fig, ax


def plot_missing_heatmap(df: pd.DataFrame):
    fig, ax = _figure()
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing values")
    fig.tight_layout()
    return fig, ax


def plot_pairplot(df: pd.DataFrame, columns: list[str] | None = None, hue: str | None = None):
    selected = df[columns] if columns is not None else df.select_dtypes(include="number")
    if hue is not None and hue not in selected.columns:
        selected = selected.copy()
        selected[hue] = df[hue]
    graph = sns.pairplot(selected, hue=hue)
    graph.fig.tight_layout()
    return graph
