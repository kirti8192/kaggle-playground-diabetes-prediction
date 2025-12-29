# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import optuna

# %%
# params
SEED = 42
TARGET = "diagnosed_diabetes"
ID_COLUMN = "id"
N_FOLDS = 5
TOP_N_IMPORTANCE = 30

TRAIN_PATH = "../data/raw/playground-series-s5e12/train.csv"
TEST_PATH = "../data/raw/playground-series-s5e12/test.csv"

SUBMISSION_DIR = "../submissions"
SUBMISSION_PREFIX = "submission"
SUBMISSION_TAG = "catboost_cv"

PLOT_AUC_CURVES = True
PLOT_FEATURE_IMPORTANCE = True
SAVE_SUBMISSION = True
EVAL_VERBOSE = 100

CAT_COLS = [
    "education_level",
    "income_level",
    "smoking_status",
    "gender",
    "ethnicity",
    "employment_status",
]

CATBOOST_NUM_ITERATIONS = 2000
CATBOOST_LEARNING_RATE = 0.1
CATBOOST_DEPTH = 5
CATBOOST_EARLY_STOPPING_ROUNDS = 100
CATBOOST_EVAL_METRIC = "AUC"
CATBOOST_LOSS_FUNCTION = "Logloss"
CATBOOST_L2_LEAF_REG = 3.0

# ---- Optuna tuning ----
RUN_OPTUNA = True           # set False to skip tuning
OPTUNA_N_TRIALS = 30        # increase later (e.g., 100+) when stable
OPTUNA_TIMEOUT_SEC = None   # or set seconds, e.g., 3600
OPTUNA_TUNE_FOLDS = 3       # use 3 folds for tuning speed; final training uses N_FOLDS
OPTUNA_SHOW_PROGRESS = True
OPTUNA_DIRECTION = "maximize"

# Use AUC as our outer-loop score (we compute with sklearn)
# For GPU training, CatBoost may not support AUC as an eval metric per-iteration; we use Logloss for early stopping.
OPTUNA_EVAL_METRIC = "Logloss"
OPTUNA_CUSTOM_METRIC = ["AUC"]

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
def build_model(seed, params=None):
    params = params or {}

    # Defaults (can be overridden by Optuna via `params`)
    iterations = params.get("iterations", CATBOOST_NUM_ITERATIONS)
    learning_rate = params.get("learning_rate", CATBOOST_LEARNING_RATE)
    depth = params.get("depth", CATBOOST_DEPTH)
    l2_leaf_reg = params.get("l2_leaf_reg", CATBOOST_L2_LEAF_REG)
    random_strength = params.get("random_strength", 1.0)
    bagging_temperature = params.get("bagging_temperature", 1.0)
    rsm = params.get("rsm", None)  # feature subsampling; None means CatBoost default
    border_count = params.get("border_count", 128)

    model_kwargs = dict(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=seed,
        early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
        # Keep training stable across CPU/GPU: use Logloss for early stopping, compute AUC externally.
        loss_function=CATBOOST_LOSS_FUNCTION,
        eval_metric=OPTUNA_EVAL_METRIC if RUN_OPTUNA else CATBOOST_EVAL_METRIC,
        custom_metric=OPTUNA_CUSTOM_METRIC if RUN_OPTUNA else None,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        border_count=border_count,
        allow_writing_files=False,
        verbose=False,  # control verbosity in fit()
    )

    # Only set rsm if provided
    if rsm is not None:
        model_kwargs["rsm"] = rsm

    return CatBoostClassifier(**model_kwargs)

# %%
def train_cv(df_train_total, df_test, seed, n_folds, target, id_column, eval_verbose, model_params=None, return_models=True):

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

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_total, y_total), start=1):

        df_train = df_train_total.iloc[train_index]
        df_val = df_train_total.iloc[val_index]

        X_train = df_train.drop(columns=[target, id_column]).copy()
        y_train = df_train[target].astype(int)

        X_val = df_val.drop(columns=[target, id_column]).copy()
        y_val = df_val[target].astype(int)

        X_test = df_test.drop(columns=[id_column]).copy()

        # get categorical feature indices for CatBoost
        cat_feature_indices = [X_train.columns.get_loc(col) for col in CAT_COLS if col in X_train.columns]

        model = build_model(seed, params=model_params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=eval_verbose,
        )

        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        print(f"Fold {fold_idx} Validation AUC: {val_auc:.4f}")

        fold_eval_results.append(model.get_evals_result())
        if return_models:
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
    }

# %%
def optuna_objective(trial, df_train_total, df_test, seed, target, id_column):

    # Search space tuned for tabular CatBoost; keep it tight for efficiency.
    params = {
        "iterations": trial.suggest_int("iterations", 1500, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),        # row subsampling
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),                                        # feature subsampling
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),          # feature binning
    }

    cv_results = train_cv(
        df_train_total=df_train_total,
        df_test=df_test,
        seed=seed,
        n_folds=OPTUNA_TUNE_FOLDS,
        target=target,
        id_column=id_column,
        eval_verbose=0,
        model_params=params,
        return_models=False,
    )
    return float(np.mean(cv_results["val_scores"]))

# %%
def run_optuna_tuning(df_train_total, df_test, seed, target, id_column):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=OPTUNA_DIRECTION, sampler=sampler)

    study.optimize(
        lambda trial: optuna_objective(trial, df_train_total, df_test, seed, target, id_column),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SEC,
        show_progress_bar=OPTUNA_SHOW_PROGRESS,
    )

    print("Optuna best value:", study.best_value)
    print("Optuna best params:", study.best_params)
    return study.best_params, study.best_value

# %%
def plot_auc_curves(fold_eval_results):
    """
    CatBoost stores eval history in a dict like:
      {
        "learn": {"Logloss": [...], "AUC": [...]},
        "validation": {"Logloss": [...], "AUC": [...]}
      }
    Metric key casing can vary; we try common variants.
    This function plots per-iteration curves for BOTH Logloss and AUC.
    """
    def _pick_metric(metrics_dict, candidates):
        for k in candidates:
            if k in metrics_dict:
                return k
        return None

    # Two separate figures for readability
    fig1 = plt.figure(figsize=(10, 6))
    for fold_idx, evals_result in enumerate(fold_eval_results, start=1):
        val_dict = evals_result.get("validation") or evals_result.get("validation_0") or {}
        logloss_key = _pick_metric(val_dict, ["Logloss", "logloss", "LogLoss", "loss"])
        if logloss_key is None:
            continue
        curve = val_dict[logloss_key]
        plt.plot(range(1, len(curve) + 1), curve, label=f"Fold {fold_idx}")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Logloss")
    plt.title("CatBoost Validation Logloss per Iteration (per fold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(10, 6))
    for fold_idx, evals_result in enumerate(fold_eval_results, start=1):
        val_dict = evals_result.get("validation") or evals_result.get("validation_0") or {}
        auc_key = _pick_metric(val_dict, ["AUC", "auc"])
        if auc_key is None:
            continue
        curve = val_dict[auc_key]
        plt.plot(range(1, len(curve) + 1), curve, label=f"Fold {fold_idx}")
    plt.xlabel("Iteration")
    plt.ylabel("Validation AUC")
    plt.title("CatBoost Validation AUC per Iteration (per fold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
def plot_feature_importance(model, feature_names, top_n):
    """
    CatBoost feature importance. We use PredictionValuesChange (default) which is
    fast and works well for debugging.
    """
    feature_importances = model.get_feature_importance()
    if len(feature_importances) != len(feature_names):
        raise ValueError(
            f"Mismatch: importances={len(feature_importances)} vs names={len(feature_names)}. "
            "This indicates a preprocessing/feature-name alignment issue."
        )

    # sort descending
    sorted_idx = np.argsort(feature_importances)[::-1]

    if top_n is not None and top_n > 0:
        top_n = min(top_n, len(sorted_idx))
        sorted_idx = sorted_idx[:top_n]

    # For barh, reverse to have the largest on top
    sorted_idx_plot = sorted_idx[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(
        range(len(sorted_idx_plot)),
        feature_importances[sorted_idx_plot],
        align="center",
    )
    plt.yticks(
        range(len(sorted_idx_plot)),
        [feature_names[i] for i in sorted_idx_plot],
    )
    plt.xlabel("Feature Importance")
    plt.title("CatBoost Feature Importance")
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
def main():
    df_train_total, df_test = load_data(TRAIN_PATH, TEST_PATH)
    df_train_total = feature_engineering(df_train_total)
    df_test = feature_engineering(df_test)

    best_params = None
    best_value = None

    if RUN_OPTUNA:
        best_params, best_value = run_optuna_tuning(
            df_train_total=df_train_total,
            df_test=df_test,
            seed=SEED,
            target=TARGET,
            id_column=ID_COLUMN,
        )

    cv_results = train_cv(
        df_train_total,
        df_test,
        SEED,
        N_FOLDS,
        TARGET,
        ID_COLUMN,
        EVAL_VERBOSE,
        model_params=best_params,
        return_models=True,
    )

    if PLOT_AUC_CURVES:
        plot_auc_curves(cv_results["fold_eval_results"])

    if PLOT_FEATURE_IMPORTANCE:
        plot_feature_importance(
            cv_results["models"][-1],
            cv_results["feature_names_per_fold"][-1],
            TOP_N_IMPORTANCE,
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
