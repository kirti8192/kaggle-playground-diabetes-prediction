"""Plot training figures from CV stats CSVs.

Expected inputs (CSV):
1) Per-iteration stats (long format):
   columns: model, fold, iteration, split (train|val), metric (auc|logloss), value
2) Per-fold summary stats:
   columns: model, fold, best_iteration, best_val_auc, best_val_logloss

This script generates:
- AUC curves (train/val) per fold + mean±std across folds
- Logloss curves (train/val) per fold + mean±std across folds
- Bar charts for best_val_auc and best_val_logloss across folds

Usage:
  python scripts/plot_figures.py \
    --stats_csv logs/cv_fold_xgboost_stats.csv \
    --summary_csv logs/cv_fold_xgboost_summary_stats.csv \
    --out_dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def load_stats(stats_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(stats_csv)
    _require_cols(df, ["model", "fold", "iteration", "split", "metric", "value"], "stats_csv")
    # normalize dtypes
    df["fold"] = df["fold"].astype(int)
    df["iteration"] = df["iteration"].astype(int)
    df["split"] = df["split"].astype(str)
    df["metric"] = df["metric"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def load_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    _require_cols(df, ["model", "fold", "best_iteration", "best_val_auc", "best_val_logloss"], "summary_csv")
    df["fold"] = df["fold"].astype(int)
    df["best_iteration"] = df["best_iteration"].astype(int)
    df["best_val_auc"] = pd.to_numeric(df["best_val_auc"], errors="coerce")
    df["best_val_logloss"] = pd.to_numeric(df["best_val_logloss"], errors="coerce")
    return df


def _pivot_metric(df: pd.DataFrame, metric: str, split: str) -> pd.DataFrame:
    """Return a wide table with index=iteration, columns=fold, values=value."""
    sub = df[(df["metric"] == metric) & (df["split"] == split)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for metric={metric!r}, split={split!r}.")

    wide = sub.pivot_table(index="iteration", columns="fold", values="value", aggfunc="mean")
    wide = wide.sort_index()
    return wide


def plot_curves_per_fold(
    df_stats: pd.DataFrame,
    metric: str,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot train+val curves on the same chart, one line per fold per split."""

    train = _pivot_metric(df_stats, metric=metric, split="train")
    val = _pivot_metric(df_stats, metric=metric, split="val")

    fig, ax = plt.subplots(figsize=(10, 6))

    folds = sorted(set(train.columns).union(set(val.columns)))
    for f in folds:
        if f in train.columns:
            ax.plot(train.index, train[f], label=f"fold {f} (train)", alpha=0.6)
        if f in val.columns:
            ax.plot(val.index, val[f], label=f"fold {f} (val)", alpha=0.9)

    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"{title_prefix}{metric} curves by fold")
    ax.grid(True, alpha=0.25)

    # keep legend readable: put outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_std(
    df_stats: pd.DataFrame,
    metric: str,
    split: str,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot mean±std across folds for a given split."""

    wide = _pivot_metric(df_stats, metric=metric, split=split)
    mean = wide.mean(axis=1)
    std = wide.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wide.index, mean, label=f"mean ({split})")
    ax.fill_between(wide.index, mean - std, mean + std, alpha=0.2, label="±1 std")

    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"{title_prefix}{metric} mean±std across folds ({split})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_bars(df_summary: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = df_summary["fold"].astype(int).tolist()

    # best AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    aucs = df_summary["best_val_auc"].to_numpy(dtype=float)
    ax.bar([str(f) for f in folds], aucs)
    ax.axhline(float(np.nanmean(aucs)), linestyle="--", linewidth=1, label=f"mean={np.nanmean(aucs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("best_val_auc")
    ax.set_title(f"{title_prefix}Best validation AUC by fold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "best_val_auc_by_fold.png", dpi=200)
    plt.close(fig)

    # best logloss
    fig, ax = plt.subplots(figsize=(8, 5))
    lls = df_summary["best_val_logloss"].to_numpy(dtype=float)
    ax.bar([str(f) for f in folds], lls)
    ax.axhline(float(np.nanmean(lls)), linestyle="--", linewidth=1, label=f"mean={np.nanmean(lls):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("best_val_logloss")
    ax.set_title(f"{title_prefix}Best validation logloss by fold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "best_val_logloss_by_fold.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CV training figures from stats CSVs")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"), help="Directory to save figures")
    parser.add_argument("--title_prefix", type=str, default="", help="Optional title prefix")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    logs_dir = (base_dir / ".." / "logs").resolve()
    stats_csv = logs_dir / "cv_fold_xgboost_stats.csv"
    summary_csv = logs_dir / "cv_fold_xgboost_summary_stats.csv"

    df_stats = load_stats(stats_csv)
    df_summary = load_summary(summary_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Curves per fold (train + val)
    plot_curves_per_fold(df_stats, metric="auc", out_path=args.out_dir / "auc_curves_by_fold.png", title_prefix=args.title_prefix)
    plot_curves_per_fold(df_stats, metric="logloss", out_path=args.out_dir / "logloss_curves_by_fold.png", title_prefix=args.title_prefix)

    # Mean ± std curves
    plot_mean_std(df_stats, metric="auc", split="val", out_path=args.out_dir / "auc_val_mean_std.png", title_prefix=args.title_prefix)
    plot_mean_std(df_stats, metric="auc", split="train", out_path=args.out_dir / "auc_train_mean_std.png", title_prefix=args.title_prefix)
    plot_mean_std(df_stats, metric="logloss", split="val", out_path=args.out_dir / "logloss_val_mean_std.png", title_prefix=args.title_prefix)
    plot_mean_std(df_stats, metric="logloss", split="train", out_path=args.out_dir / "logloss_train_mean_std.png", title_prefix=args.title_prefix)

    # Best bars
    plot_best_bars(df_summary, out_dir=args.out_dir, title_prefix=args.title_prefix)

    print(f"Saved figures to: {args.out_dir.resolve()}")
    print(f"Loaded stats from: {logs_dir}")


if __name__ == "__main__":
    main()
