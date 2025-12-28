from sklearn.model_selection import cross_validate
from sklearn.utils.multiclass import type_of_target
import numpy as np


def cross_validate_classification(model, X, y, cv=5, n_jobs=-1, verbose=True, return_train=False):
    """
    Cross-validate a classifier with accuracy, precision, recall, F1, ROC-AUC, and log-loss.
    Auto-detects binary vs multi-class.
    """
    # Detect binary vs multi-class
    target_type = type_of_target(y)
    is_binary = target_type == "binary"

    if is_binary:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "neg_log_loss": "neg_log_loss",
        }
    else:
        scoring = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
            "roc_auc_ovr": "roc_auc_ovr",
            "neg_log_loss": "neg_log_loss",
        }

    cv_res = cross_validate(
        model, X, y, cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=return_train
    )

    # Summarize mean ± std for each metric
    summary = {}
    for key in scoring:
        sk_key = f"test_{key}"
        mean, std = np.mean(cv_res[sk_key]), np.std(cv_res[sk_key])
        if key == "neg_log_loss":
            mean, std = -mean, std  # flip sign
            label = "log_loss"
        else:
            label = key
        summary[label] = (mean, std)

    if verbose:
        print("Cross-validation results:")
        for metric, (mean, std) in summary.items():
            print(f"  {metric:12s}: {mean:.3f} ± {std:.3f}")

    return cv_res, summary


def cross_validate_regression(model, X, y, cv=5, n_jobs=-1, verbose=True, return_train=False):
    """
    Cross-validate a regressor with RMSE, MAE, MSE, R^2, and explained variance metrics.
    """
    scoring = {
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "neg_mean_squared_error": "neg_mean_squared_error",
        "r2": "r2",
        "explained_variance": "explained_variance",
    }

    cv_res = cross_validate(
        model, X, y, cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=return_train
    )

    summary = {}
    display_labels = {
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_absolute_error": "mae",
        "neg_mean_squared_error": "mse",
    }
    for key in scoring:
        sk_key = f"test_{key}"
        mean, std = np.mean(cv_res[sk_key]), np.std(cv_res[sk_key])
        if key.startswith("neg_"):
            mean = -mean
        label = display_labels.get(key, key)
        summary[label] = (mean, std)

    if verbose:
        print("Cross-validation results:")
        for metric, (mean, std) in summary.items():
            print(f"  {metric:12s}: {mean:.3f} ± {std:.3f}")

    return cv_res, summary
