# %%
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# %%
# params
SEED = 42
TARGET = "diagnosed_diabetes"
ID_COLUMN = "id"
N_FOLDS = 5

TOP_N_IMPORTANCE = 30

# categorical columns (native LightGBM categorical handling)
CAT_COLS = [
    "education_level",
    "income_level",
    "smoking_status",
    "gender",
    "ethnicity",
    "employment_status",
]

# toggles
PLOT_AUC_CURVES = True
PLOT_FEATURE_IMPORTANCE = True
SAVE_SUBMISSION = True

SAVE_LOGS = True
LOG_DIR = "../logs"
LOG_BASENAME = "cv_fold_lightgbm"

SAVE_MODEL = True
MODEL_DIR = "../models"
MODEL_FILENAME = "lightgbm_optuna.txt"

TRAIN_PATH = "../data/raw/playground-series-s5e12/train.csv"
TEST_PATH = "../data/raw/playground-series-s5e12/test.csv"

SUBMISSION_DIR = "../submissions"
SUBMISSION_PREFIX = "submission"
SUBMISSION_TAG = "lgbm_cv"

LGBM_PARAMS_PATH = "../parameters/lightgbm_optuna.json"

LGBM_EARLY_STOPPING_ROUNDS = 200
LGBM_LOG_EVAL_PERIOD = 0  # set >0 to log every N rounds
LGBM_VERBOSITY = -1

LGBM_REQUIRED_KEYS = [
    "LGBM_N_ESTIMATORS",
    "LGBM_LEARNING_RATE",
    "LGBM_NUM_LEAVES",
    "LGBM_MAX_DEPTH",
    "LGBM_MIN_CHILD_SAMPLES",
    "LGBM_SUBSAMPLE",
    "LGBM_COLSAMPLE_BYTREE",
    "LGBM_REG_ALPHA",
    "LGBM_REG_LAMBDA",
]

# %%
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# %%
def feature_engineering(df):
    eps = 1e-6
    df = df.copy()
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["non_hdl"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["ldl_hdl"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + eps)
    df["tg_hdl"] = df["triglycerides"] / (df["hdl_cholesterol"] + eps)
    df["total_hdl"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + eps)
    df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + eps)
    return df

# %%
def split_features_target(df, target, id_column):
    X = df.drop(columns=[target, id_column]).copy()
    y = df[target].astype(int)
    return X, y

# %%
def load_lgbm_params(path):
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    missing = [k for k in LGBM_REQUIRED_KEYS if k not in params]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing keys in {path}: {missing_str}")
    return params

# %%
def build_model(seed, params):
    return LGBMClassifier(
        objective="binary",
        n_estimators=params["LGBM_N_ESTIMATORS"],
        learning_rate=params["LGBM_LEARNING_RATE"],
        num_leaves=params["LGBM_NUM_LEAVES"],
        max_depth=params["LGBM_MAX_DEPTH"],
        min_child_samples=params["LGBM_MIN_CHILD_SAMPLES"],
        subsample=params["LGBM_SUBSAMPLE"],
        colsample_bytree=params["LGBM_COLSAMPLE_BYTREE"],
        reg_alpha=params["LGBM_REG_ALPHA"],
        reg_lambda=params["LGBM_REG_LAMBDA"],
        random_state=seed,
        n_jobs=-1,
        verbosity=LGBM_VERBOSITY,
    )

# %%
def collect_lgbm_logs(evals_result, fold_idx):
    records = []
    split_map = {
        "train": "train",
        "val": "val",
    }
    metric_map = {
        "binary_logloss": "logloss",
        "auc": "auc",
    }
    for split_key, split_label in split_map.items():
        if split_key not in evals_result:
            continue
        for metric_key, metric_label in metric_map.items():
            values = evals_result[split_key].get(metric_key)
            if values is None:
                continue
            for iteration, value in enumerate(values, start=1):
                records.append(
                    {
                        "model": "lightgbm",
                        "fold": fold_idx,
                        "iteration": iteration,
                        "split": split_label,
                        "metric": metric_label,
                        "value": value,
                    }
                )
    return records

# %%
def get_lgbm_best_iteration(model, evals_result):
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is not None and best_iteration > 0:
        return int(best_iteration)
    val_metrics = evals_result.get("val", {})
    values = val_metrics.get("auc") or val_metrics.get("binary_logloss")
    if values:
        return len(values)
    return None

# %%
def get_lgbm_best_metrics(evals_result, best_iteration):
    if best_iteration is None:
        return None, None
    idx = best_iteration - 1
    val_metrics = evals_result.get("val", {})
    auc_values = val_metrics.get("auc")
    logloss_values = val_metrics.get("binary_logloss")
    best_val_auc = auc_values[idx] if auc_values and idx < len(auc_values) else None
    best_val_logloss = logloss_values[idx] if logloss_values and idx < len(logloss_values) else None
    return best_val_auc, best_val_logloss

# %%
def train_cv(df_train_total, df_test, seed, n_folds, target, id_column, eval_verbose, model_params):

    X_total, y_total = split_features_target(df_train_total, target, id_column)
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_eval_results = []
    models = []
    feature_names_per_fold = []
    test_pred_folds = []
    val_scores = []
    log_records = []
    best_records = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_total, y_total), start=1):

        df_train = df_train_total.iloc[train_index]
        df_val = df_train_total.iloc[val_index]

        X_train = df_train.drop(columns=[target, id_column]).copy()
        y_train = df_train[target].astype(int)

        X_val = df_val.drop(columns=[target, id_column]).copy()
        y_val = df_val[target].astype(int)

        X_test = df_test.drop(columns=[id_column]).copy()

        # LightGBM: use native categorical handling by setting dtype=category
        for c in CAT_COLS:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype("category")
                X_val[c] = X_val[c].astype("category")
                X_test[c] = X_test[c].astype("category")

        model = build_model(seed, model_params)
        callbacks = [
            __import__("lightgbm").early_stopping(
                stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
                verbose=False,
            ),
        ]
        if eval_verbose and eval_verbose > 0:
            callbacks.append(__import__("lightgbm").log_evaluation(period=eval_verbose))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val), (X_train, y_train)],
            eval_names=["val", "train"],
            eval_metric=["auc", "binary_logloss"],
            categorical_feature=[c for c in CAT_COLS if c in X_train.columns],
            callbacks=callbacks,
        )

        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        print(f"Fold {fold_idx} Validation AUC: {val_auc:.4f}")

        evals_result = model.evals_result_
        fold_eval_results.append(evals_result)
        log_records.extend(collect_lgbm_logs(evals_result, fold_idx))
        best_iteration = get_lgbm_best_iteration(model, evals_result)
        best_val_auc, best_val_logloss = get_lgbm_best_metrics(evals_result, best_iteration)
        if best_iteration is not None:
            best_records.append(
                {
                    "model": "lightgbm",
                    "fold": fold_idx,
                    "best_iteration": best_iteration,
                    "best_val_auc": best_val_auc,
                    "best_val_logloss": best_val_logloss,
                }
            )
        models.append(model)
        feature_names_per_fold.append(X_train.columns.tolist())
        test_pred_folds.append(model.predict_proba(X_test)[:, 1])
        val_scores.append(val_auc)

    return {
        "models": models,
        "feature_names_per_fold": feature_names_per_fold,
        "fold_eval_results": fold_eval_results,
        "test_pred_folds": test_pred_folds,
        "val_scores": val_scores,
        "log_records": log_records,
        "best_records": best_records,
    }

# %%
def plot_auc_curves(fold_eval_results):
    """
    LightGBM stores eval history in a dict like:
      {
        "training": {"auc": [...], "binary_logloss": [...]},
        "valid_0": {"auc": [...], "binary_logloss": [...]}
      }
    This plots per-iteration curves for BOTH binary_logloss and auc on the validation set.
    """
    def _pick(metrics_dict, candidates):
        for k in candidates:
            if k in metrics_dict:
                return k
        return None

    # Logloss
    plt.figure(figsize=(10, 6))
    for fold_idx, evals_result in enumerate(fold_eval_results, start=1):
        val_dict = evals_result.get("valid_0", {})
        loss_key = _pick(val_dict, ["binary_logloss", "logloss", "l2"])
        if loss_key is None:
            continue
        curve = val_dict[loss_key]
        plt.plot(range(1, len(curve) + 1), curve, label=f"Fold {fold_idx}")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Logloss")
    plt.title("LightGBM Validation Logloss per Iteration (per fold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # AUC
    plt.figure(figsize=(10, 6))
    for fold_idx, evals_result in enumerate(fold_eval_results, start=1):
        val_dict = evals_result.get("valid_0", {})
        auc_key = _pick(val_dict, ["auc"])
        if auc_key is None:
            continue
        curve = val_dict[auc_key]
        plt.plot(range(1, len(curve) + 1), curve, label=f"Fold {fold_idx}")
    plt.xlabel("Iteration")
    plt.ylabel("Validation AUC")
    plt.title("LightGBM Validation AUC per Iteration (per fold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
def plot_feature_importance(model, feature_names, top_n):
    """
    LightGBM feature importance from the sklearn API.
    """
    feature_importances = model.feature_importances_
    if len(feature_importances) != len(feature_names):
        raise ValueError(
            f"Mismatch: importances={len(feature_importances)} vs names={len(feature_names)}. "
            "This indicates a preprocessing/feature-name alignment issue."
        )

    sorted_idx = np.argsort(feature_importances)[::-1]
    if top_n is not None and top_n > 0:
        top_n = min(top_n, len(sorted_idx))
        sorted_idx = sorted_idx[:top_n]

    sorted_idx_plot = sorted_idx[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx_plot)), feature_importances[sorted_idx_plot], align="center")
    plt.yticks(range(len(sorted_idx_plot)), [feature_names[i] for i in sorted_idx_plot])
    plt.xlabel("Feature Importance")
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.show()

# %%
def save_submission(df_test, y_pred, id_column, target, submission_dir, submission_prefix, submission_tag):
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{submission_tag}" if submission_tag else ""
    filename = f"{submission_prefix}_{timestamp}{suffix}.csv"
    path = submission_dir / filename

    submission = pd.DataFrame(
        {
            id_column: df_test[id_column],
            target: y_pred,
        }
    )
    submission.to_csv(path, index=False)
    print(f"Saved: {path}")

# %%
def save_model(model, model_dir, filename):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    model.booster_.save_model(str(path))
    print(f"Saved: {path}")

# %%
def save_logs(log_records, best_records, log_dir, log_basename):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_records:
        log_df = pd.DataFrame(log_records)
        log_df = log_df[["model", "fold", "iteration", "split", "metric", "value"]]
        log_df.to_csv(log_dir / f"{log_basename}.csv", index=False)
    if best_records:
        best_df = pd.DataFrame(best_records)
        best_df = best_df[["model", "fold", "best_iteration", "best_val_auc", "best_val_logloss"]]
        best_df.to_csv(log_dir / f"{log_basename}_summary.csv", index=False)

# %%
def main():
    df_train_total, df_test = load_data(TRAIN_PATH, TEST_PATH)
    df_train_total = feature_engineering(df_train_total)
    df_test = feature_engineering(df_test)

    model_params = load_lgbm_params(LGBM_PARAMS_PATH)
    cv_results = train_cv(
        df_train_total,
        df_test,
        SEED,
        N_FOLDS,
        TARGET,
        ID_COLUMN,
        LGBM_LOG_EVAL_PERIOD,
        model_params,
    )

    if PLOT_AUC_CURVES:
        plot_auc_curves(cv_results["fold_eval_results"])

    if PLOT_FEATURE_IMPORTANCE:
        plot_feature_importance(
            cv_results["models"][-1],
            cv_results["feature_names_per_fold"][-1],
            TOP_N_IMPORTANCE,
        )

    if SAVE_LOGS:
        save_logs(
            cv_results["log_records"],
            cv_results["best_records"],
            LOG_DIR,
            LOG_BASENAME,
        )

    if SAVE_MODEL:
        save_model(
            cv_results["models"][-1],
            MODEL_DIR,
            MODEL_FILENAME,
        )

    if SAVE_SUBMISSION:
        y_test_pred = np.mean(np.vstack(cv_results["test_pred_folds"]), axis=0)
        save_submission(
            df_test,
            y_test_pred,
            ID_COLUMN,
            TARGET,
            SUBMISSION_DIR,
            SUBMISSION_PREFIX,
            SUBMISSION_TAG,
        )

if __name__ == "__main__":
    main()

# %%
