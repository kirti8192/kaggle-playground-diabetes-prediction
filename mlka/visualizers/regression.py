"""Model visualizations for regression tasks."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error


def visualize_linear_regression(y_true, y_pred):
    """Plot standard diagnostics for regression predictions."""
    rmse = root_mean_squared_error(y_true, y_pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bounds = (min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(bounds, bounds, linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual (Validation)  |  RMSE: {rmse:.0f}")
    plt.tight_layout()
    plt.show()

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, edgecolor="k")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.show()

    eps = 1e-12
    rel_err = residuals / np.maximum(np.abs(y_true), eps)
    plt.figure(figsize=(6, 4))
    plt.hist(rel_err, bins=50, edgecolor="k")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Relative Error (residual / |actual|)")
    plt.ylabel("Count")
    plt.title("Relative Error Histogram")
    plt.tight_layout()
    plt.show()
