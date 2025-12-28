"""Utilities for tabular dataset exploration."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def _print_section(title: str) -> None:
    print("=" * 40)
    print(title)
    print("=" * 40)


def peek_df(df: DataFrame, target_col: Optional[str] = None, max_cols: int = 10) -> None:
    """Quickly inspect key properties of a DataFrame."""
    _print_section("SHAPE OF DATAFRAME")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    _print_section("COLUMN PREVIEW")
    cols = df.columns.tolist()
    if len(cols) > max_cols:
        print(cols[:max_cols], "...")
    else:
        print(cols)
    print()

    _print_section("HEAD OF DATAFRAME (truncated)")
    display_cols = cols[:max_cols]
    print(df[display_cols].head(), "\n")

    _print_section("INFO")
    df.info()
    print()

    _print_section("MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            "Missing Values": missing,
            "Percent": missing_percent.round(2),
        })
        print(missing_df, "\n")
    else:
        print("No missing values.\n")

    _print_section("DESCRIPTIVE STATS (NUMERIC)")
    print(df.describe().T.iloc[:max_cols], "\n")

    _print_section("DESCRIPTIVE STATS (CATEGORICAL)")
    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe().T.iloc[:max_cols], "\n")
    else:
        print("No categorical (object) columns to describe.\n")

    _print_section("UNIQUE VALUES (small cardinality)")
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 5:
            print(f"{col}: {df[col].unique()}")

    _print_section("FEATURE DISTRIBUTIONS (NUMERIC)")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) > max_cols:
        print(f"Too many numeric columns ({len(num_cols)}). Showing first {max_cols}.")
        num_cols = num_cols[:max_cols]

    if num_cols:
        df[num_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Feature Distributions", y=1.02)
        plt.show()
    else:
        print("No numeric columns available for histograms.\n")

    if target_col:
        _print_section(f"TARGET COLUMN DISTRIBUTION: {target_col}")
        print(df[target_col].value_counts(), "\n")

        sns.countplot(x=target_col, data=df)
        plt.title(f"Countplot of '{target_col}'")
        plt.show()

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] <= 20 and not numeric_df.empty:
        _print_section("CORRELATION HEATMAP (NUMERIC FEATURES)")
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
    elif numeric_df.shape[1] > 20:
        _print_section("CORRELATION HEATMAP (NUMERIC FEATURES)")
        print("Too many numeric columns for correlation heatmap. Skipping plot.\n")
