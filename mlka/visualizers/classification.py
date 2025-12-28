"""Model visualizations for classification tasks."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def _plot_binary_curves(y_true_bin, y_score, subtitle="") -> None:
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve{subtitle}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve{subtitle}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_logistic_regression(y_true, y_pred, y_proba, *, class_names=None):
    """Visualize confusion matrix, ROC/PR curves, and probability histograms."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_proba)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    print(cm)

    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes == 2:
        if y_score.ndim == 2:
            if y_score.shape[1] != 2:
                raise ValueError("For binary classification, provide probabilities as shape (n,) or (n, 2).")
            pos_proba = y_score[:, 1]
        else:
            pos_proba = y_score
        _plot_binary_curves(y_true == classes[1], pos_proba)

        plt.figure(figsize=(6, 4))
        plt.hist(pos_proba[y_true == classes[0]], bins=30, alpha=0.6, label=f"Class {classes[0]}")
        plt.hist(pos_proba[y_true == classes[1]], bins=30, alpha=0.6, label=f"Class {classes[1]}")
        plt.xlabel("Predicted Probability (positive class)")
        plt.ylabel("Count")
        plt.title("Predicted Probability Distribution (Binary)")
        plt.legend()
    else:
        if y_score.ndim != 2 or y_score.shape[1] != n_classes:
            raise ValueError("For multiclass classification, y_proba must be shape (n_samples, n_classes).")

        y_true_bin = label_binarize(y_true, classes=classes)
        _plot_binary_curves(y_true_bin.ravel(), y_score.ravel(), subtitle=" (micro-avg)")

        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        true_indices = np.array([class_to_index[val] for val in y_true])
        max_proba = y_score.max(axis=1)
        true_class_proba = y_score[np.arange(len(y_score)), true_indices]
        plt.figure(figsize=(6, 4))
        plt.hist(max_proba, bins=30, alpha=0.6, label="Max predicted prob")
        plt.hist(true_class_proba, bins=30, alpha=0.6, label="True-class prob")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.title("Predicted Probability Distribution (Multiclass)")
        plt.legend()
    plt.tight_layout()
    plt.show()
