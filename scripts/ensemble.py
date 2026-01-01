# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# %%
# params
SEED = 42
TARGET = "diagnosed_diabetes"
ID_COLUMN = "id"

PRED_DIR = "../predictions"
FIGURE_DIR = "../figures"
SUBMISSION_DIR = "../submissions"

OOF_FILES = {
    "xgboost": "../predictions/oof_xgboost.csv",
    "lightgbm": "../predictions/oof_lightgbm.csv",
    "catboost": "../predictions/oof_catboost.csv",
}

TEST_FILES = {
    "xgboost": "../predictions/test_xgboost.csv",
    "lightgbm": "../predictions/test_lightgbm.csv",
    "catboost": "../predictions/test_catboost.csv",
}

SAVE_SUBMISSION = True
SUBMISSION_PREFIX = "submission"
SUBMISSION_TAG = "ensemble"

TUNE_WEIGHTS = True
GRID_STEP = 0.05
MANUAL_WEIGHTS = {
    "xgboost": 0.33,
    "lightgbm": 0.34,
    "catboost": 0.33,
}


# %%
def load_oof(path, model_name):
    df = pd.read_csv(path)
    expected = {ID_COLUMN, "y_true", "oof_pred"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        raise ValueError(f"{model_name} OOF missing columns: {sorted(missing)}")
    df = df[[ID_COLUMN, "y_true", "oof_pred"]].copy()
    df = df.rename(columns={"oof_pred": f"pred_{model_name}"})
    return df


# %%
def load_test(path, model_name):
    df = pd.read_csv(path)
    expected = {ID_COLUMN, "test_pred"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        raise ValueError(f"{model_name} test missing columns: {sorted(missing)}")
    df = df[[ID_COLUMN, "test_pred"]].copy()
    df = df.rename(columns={"test_pred": f"pred_{model_name}"})
    return df


# %%
def merge_oof(oof_files):
    dfs = []
    for model_name, path in oof_files.items():
        dfs.append(load_oof(path, model_name))
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=ID_COLUMN, how="inner")
    y_true_cols = [col for col in merged.columns if col.startswith("y_true")]
    if not y_true_cols:
        raise ValueError("Missing y_true in merged OOF data")
    base = merged[y_true_cols[0]]
    for col in y_true_cols[1:]:
        if not base.equals(merged[col]):
            raise ValueError("y_true mismatch across OOF files")
    if "y_true" not in merged.columns:
        merged = merged.rename(columns={y_true_cols[0]: "y_true"})
        y_true_cols = [col for col in merged.columns if col.startswith("y_true")]
    extra = [col for col in y_true_cols if col != "y_true"]
    if extra:
        merged = merged.drop(columns=extra)
    return merged


# %%
def merge_test(test_files):
    dfs = []
    for model_name, path in test_files.items():
        dfs.append(load_test(path, model_name))
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=ID_COLUMN, how="inner")
    return merged


# %%
def tune_weights(oof_df, model_names, step):
    preds = oof_df[[f"pred_{m}" for m in model_names]].to_numpy()
    y_true = oof_df["y_true"].to_numpy()
    best_auc = -1.0
    best_weights = None

    grid = np.arange(0.0, 1.0 + 1e-8, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            weights = np.array([w1, w2, w3])
            if np.isclose(weights.sum(), 0.0):
                continue
            blended = preds @ weights
            auc = roc_auc_score(y_true, blended)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights

    return best_weights, best_auc


# %%
def save_figure(fig, figure_dir, filename):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")


# %%
def plot_auc_bar(auc_map, figure_dir):
    fig = plt.figure(figsize=(8, 5))
    labels = list(auc_map.keys())
    values = [auc_map[k] for k in labels]
    plt.bar(labels, values, color="#4c78a8")
    plt.ylabel("OOF ROC-AUC")
    plt.title("OOF ROC-AUC by Model")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    save_figure(fig, figure_dir, "ensemble_auc_bar.png")
    plt.show()


# %%
def plot_pred_correlation(oof_df, model_names, figure_dir):
    preds = oof_df[[f"pred_{m}" for m in model_names]].to_numpy()
    corr = np.corrcoef(preds.T)
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.yticks(range(len(model_names)), model_names)
    plt.title("OOF Prediction Correlation")
    plt.tight_layout()
    save_figure(fig, figure_dir, "ensemble_pred_correlation.png")
    plt.show()


# %%
def plot_pred_hist(oof_df, model_names, figure_dir):
    fig = plt.figure(figsize=(8, 5))
    bins = 40
    for model_name in model_names:
        plt.hist(
            oof_df[f"pred_{model_name}"],
            bins=bins,
            alpha=0.4,
            label=model_name,
            density=True,
        )
    plt.xlabel("OOF Prediction")
    plt.ylabel("Density")
    plt.title("OOF Prediction Distributions")
    plt.legend()
    plt.tight_layout()
    save_figure(fig, figure_dir, "ensemble_pred_hist.png")
    plt.show()


# %%
oof_df = merge_oof(OOF_FILES)
test_df = merge_test(TEST_FILES)

model_names = list(OOF_FILES.keys())

auc_map = {}
for model_name in model_names:
    auc_map[model_name] = roc_auc_score(oof_df["y_true"], oof_df[f"pred_{model_name}"])

if TUNE_WEIGHTS:
    weights, best_auc = tune_weights(oof_df, model_names, GRID_STEP)
    weight_map = dict(zip(model_names, weights))
    auc_map["ensemble"] = best_auc
else:
    weight_map = MANUAL_WEIGHTS
    weights = np.array([weight_map[m] for m in model_names])
    blended = oof_df[[f"pred_{m}" for m in model_names]].to_numpy() @ weights
    auc_map["ensemble"] = roc_auc_score(oof_df["y_true"], blended)

oof_preds = oof_df[[f"pred_{m}" for m in model_names]].to_numpy() @ weights
test_preds = test_df[[f"pred_{m}" for m in model_names]].to_numpy() @ weights

pred_dir = Path(PRED_DIR)
pred_dir.mkdir(parents=True, exist_ok=True)

oof_out = pred_dir / "oof_ensemble.csv"
oof_df_out = pd.DataFrame(
    {
        ID_COLUMN: oof_df[ID_COLUMN],
        "y_true": oof_df["y_true"],
        "oof_pred": oof_preds,
    }
)
oof_df_out.to_csv(oof_out, index=False)
print(f"Saved: {oof_out}")

test_out = pred_dir / "test_ensemble.csv"
test_df_out = pd.DataFrame(
    {
        ID_COLUMN: test_df[ID_COLUMN],
        "test_pred": test_preds,
    }
)
test_df_out.to_csv(test_out, index=False)
print(f"Saved: {test_out}")

plot_auc_bar(auc_map, FIGURE_DIR)
plot_pred_correlation(oof_df, model_names, FIGURE_DIR)
plot_pred_hist(oof_df, model_names, FIGURE_DIR)

if SAVE_SUBMISSION:
    submission_dir = Path(SUBMISSION_DIR)
    submission_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SUBMISSION_PREFIX}_{timestamp}_{SUBMISSION_TAG}.csv"
    path = submission_dir / filename
    submission = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET: test_preds})
    submission.to_csv(path, index=False)
    print(f"Saved: {path}")
# %%
