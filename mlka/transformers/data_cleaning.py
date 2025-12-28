"""Transformers for cleaning tabular data."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "FixNulls",
    "DropHighlyMissingColumns",
]

DEFAULT_NULL_VALUES: Sequence[str] = ("nan", "NaN", "NAN", "NULL", "", " ", "NA")


def _ensure_dataframe(obj):
    if not isinstance(obj, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    return obj


class FixNulls(BaseEstimator, TransformerMixin):
    """Replace custom null-like tokens with ``np.nan``."""

    def __init__(self, values: Optional[Iterable[str]] = None):
        self.values = tuple(values) if values is not None else DEFAULT_NULL_VALUES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Input must be a pandas DataFrame or Series")
        return X.replace(self.values, np.nan)


class DropHighlyMissingColumns(BaseEstimator, TransformerMixin):
    """Drop columns whose missing ratio exceeds ``thresh_percent``."""

    def __init__(self, thresh_percent: float = 0.1):
        self.thresh_percent = float(thresh_percent)
        self.cols_to_keep_: Optional[List[str]] = None

    def fit(self, X, y=None):
        df = _ensure_dataframe(X)
        missing_frac = df.isna().mean()
        self.cols_to_keep_ = [col for col in df.columns if missing_frac[col] <= self.thresh_percent]
        return self

    def transform(self, X):
        if self.cols_to_keep_ is None:
            raise ValueError("The transformer is not fitted. Call 'fit' before 'transform'.")
        df = _ensure_dataframe(X)
        return df[self.cols_to_keep_].copy()
