"""Plot training figures from CV stats CSVs.

Expected inputs (CSV):
The script auto-discovers multiple models using files named like:
  - cv_fold_<model>_stats.csv
  - cv_fold_<model>_summary_stats.csv

This script generates:
- comparison_auc_val_mean_std.png
- comparison_logloss_val_mean_std.png

Usage:
  python scripts/plot_figures.py \
    --out_dir figures \
    --logs_dir ../logs

The script scans the logs directory for model stats and summary CSVs.
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
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()

    # Normalize common aliases / library-specific labels
    def _norm_split(s: str) -> str:
        s = s.strip().lower()

        # LightGBM often uses valid_0 / valid_1, CatBoost uses learn/validation
        if s.startswith("valid") or s.startswith("val") or s.startswith("eval"):
            return "val"
        if s.startswith("train") or s in {"learn", "tr", "training"}:
            return "train"
        if s.startswith("test"):
            return "val"  # treat as eval set for plotting
        return s

    df["split"] = df["split"].map(_norm_split)

    metric_map = {
        "auc": "auc",
        "roc_auc": "auc",
        "roc-auc": "auc",
        "logloss": "logloss",
        "log_loss": "logloss",
        "log-loss": "logloss",
        "binary_logloss": "logloss",
        "cross_entropy": "logloss",
    }
    df["metric"] = df["metric"].map(lambda m: metric_map.get(m, m))

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


def _model_from_stats_filename(path: Path) -> str:
    """Extract model name from a filename like cv_fold_<model>_stats.csv"""
    name = path.name
    prefix = "cv_fold_"
    suffix = "_stats.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    # fallback: stem without extension
    return path.stem


def _model_from_summary_filename(path: Path) -> str:
    """Extract model name from a filename like cv_fold_<model>_summary_stats.csv"""
    name = path.name
    prefix = "cv_fold_"
    suffix = "_summary_stats.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def discover_log_files(logs_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (model, stats_csv, summary_csv) discovered under logs_dir."""
    stats_files = sorted(logs_dir.glob("cv_fold_*_stats.csv"))
    out: list[tuple[str, Path, Path]] = []
    for s in stats_files:
        model = _model_from_stats_filename(s)
        summary = logs_dir / f"cv_fold_{model}_summary_stats.csv"
        if summary.exists():
            out.append((model, s, summary))
    if not out:
        raise FileNotFoundError(
            f"No model log pairs found in {logs_dir}. Expected files like cv_fold_<model>_stats.csv and cv_fold_<model>_summary_stats.csv"
        )
    return out


def _pivot_metric(df: pd.DataFrame, metric: str, split: str, model: str | None = None, allow_empty: bool = False) -> pd.DataFrame:
    """Return a wide table with index=iteration, columns=fold, values=value."""
    sub = df.copy()
    if model is not None:
        sub = sub[sub["model"] == model]
    sub = sub[(sub["metric"] == metric) & (sub["split"] == split)].copy()
    if sub.empty:
        if allow_empty:
            return pd.DataFrame()
        raise ValueError(f"No rows found for metric={metric!r}, split={split!r}, model={model!r}.")
    wide = sub.pivot_table(index="iteration", columns="fold", values="value", aggfunc="mean")
    wide = wide.sort_index()
    return wide


def plot_mean_std(
    df_stats: pd.DataFrame,
    metric: str,
    split: str,
    out_path: Path,
    title_prefix: str = "",
    model: str | None = None,
) -> None:
    """Plot mean±std across folds for a given split."""

    wide = _pivot_metric(df_stats, metric=metric, split=split, model=model, allow_empty=True)
    if wide.empty:
        raise ValueError(f"No rows found for metric={metric!r}, split={split!r}, model={model!r}.")
    mean = wide.mean(axis=1)
    std = wide.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    if model:
        label = f"{model} mean ({split})"
    else:
        label = f"mean ({split})"
    ax.plot(wide.index, mean, label=label)
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


def plot_curves_per_fold(
    df_stats: pd.DataFrame,
    metric: str,
    out_path: Path,
    title_prefix: str = "",
    model: str | None = None,
) -> None:
    """Plot train+val curves on the same chart, one line per fold per split."""

    train = _pivot_metric(df_stats, metric=metric, split="train", model=model, allow_empty=True)
    val = _pivot_metric(df_stats, metric=metric, split="val", model=model, allow_empty=True)

    if val.empty and train.empty:
        raise ValueError(f"No data to plot for metric={metric!r}, model={model!r}.")

    train_cols = set(train.columns) if not train.empty else set()
    val_cols = set(val.columns) if not val.empty else set()
    folds = sorted(train_cols.union(val_cols))

    fig, ax = plt.subplots(figsize=(10, 6))

    for f in folds:
        if f in train.columns:
            label = f"{model} fold {f} (train)" if model else f"fold {f} (train)"
            ax.plot(train.index, train[f], label=label, alpha=0.6)
        if f in val.columns:
            label = f"{model} fold {f} (val)" if model else f"fold {f} (val)"
            ax.plot(val.index, val[f], label=label, alpha=0.9)

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


def plot_best_bars(df_summary: pd.DataFrame, out_dir: Path, title_prefix: str = "", model: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = df_summary["fold"].astype(int).tolist()

    prefix = title_prefix or (f"{model} – " if model else "")

    # best AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    aucs = df_summary["best_val_auc"].to_numpy(dtype=float)
    ax.bar([str(f) for f in folds], aucs)
    ax.axhline(float(np.nanmean(aucs)), linestyle="--", linewidth=1, label=f"mean={np.nanmean(aucs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("best_val_auc")
    ax.set_title(f"{prefix}Best validation AUC by fold")
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
    ax.set_title(f"{prefix}Best validation logloss by fold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "best_val_logloss_by_fold.png", dpi=200)
    plt.close(fig)


def plot_compare_models_mean_std(
    all_stats: pd.DataFrame,
    models: list[str],
    metric: str,
    split: str,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot mean ± 1 std across folds for each model on the same axes."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        wide = _pivot_metric(all_stats, metric=metric, split=split, model=model, allow_empty=True)
        if wide.empty:
            continue
        mean = wide.mean(axis=1, skipna=True)
        std = wide.std(axis=1, skipna=True)

        mask = mean.notna() & std.notna()
        x = mean.index[mask]
        y = mean[mask]
        s = std[mask]

        ax.plot(x, y, label=model)
        ax.fill_between(x, y - s, y + s, alpha=0.2)

    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"{title_prefix}{metric} mean±std across folds ({split}) – all models")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CV training figures from stats CSVs")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"), help="Directory to save figures")
    parser.add_argument("--title_prefix", type=str, default="", help="Optional title prefix")
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=None,
        help="Logs directory (defaults to ../logs relative to this script)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    logs_dir = (base_dir / ".." / "logs").resolve() if args.logs_dir is None else args.logs_dir.resolve()
    pairs = discover_log_files(logs_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats_list = []

    for model, stats_csv, _ in pairs:
        df_stats = load_stats(stats_csv)
        # Overwrite/ensure model column
        df_stats["model"] = model
        all_stats_list.append(df_stats)

    all_stats = pd.concat(all_stats_list, ignore_index=True)
    models = [model for model, _, _ in pairs]

    plot_compare_models_mean_std(
        all_stats,
        models=models,
        metric="auc",
        split="val",
        out_path=args.out_dir / "comparison_auc_val_mean_std.png",
        title_prefix=args.title_prefix,
    )

    plot_compare_models_mean_std(
        all_stats,
        models=models,
        metric="logloss",
        split="val",
        out_path=args.out_dir / "comparison_logloss_val_mean_std.png",
        title_prefix=args.title_prefix,
    )

    models_str = ", ".join(models)
    print(f"Saved figures to: {args.out_dir.resolve()}")
    print(f"Loaded stats from: {logs_dir}")
    print(f"Discovered models: {models_str}")


if __name__ == "__main__":
    main()
