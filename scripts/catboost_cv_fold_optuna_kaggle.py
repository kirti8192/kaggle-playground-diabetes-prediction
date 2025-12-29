# %%
from datetime import datetime
from pathlib import Path

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

# Kaggle: inputs are mounted under /kaggle/input, outputs must go to /kaggle/working
# If you don't know the exact competition folder name, we auto-detect train.csv/test.csv under /kaggle/input.
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"

SUBMISSION_DIR = "/kaggle/working"
SUBMISSION_PREFIX = "submission"
SUBMISSION_TAG = "catboost_cv_optuna"

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
OPTUNA_N_TRIALS = 100        # increase later (e.g., 100+) when stable
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

def resolve_kaggle_paths(train_path, test_path):
    """
    Kaggle mounts datasets under /kaggle/input/<dataset_slug>/...
    This resolves train/test CSV paths by:
      1) Using provided paths if they exist
      2) Otherwise searching /kaggle/input/**/train.csv and test.csv
    """
    train_p = Path(train_path)
    test_p = Path(test_path)

    if train_p.exists() and test_p.exists():
        return str(train_p), str(test_p)

    # Auto-detect by searching Kaggle input
    candidates_train = sorted(Path("/kaggle/input").rglob("train.csv"))
    candidates_test = sorted(Path("/kaggle/input").rglob("test.csv"))

    if not candidates_train or not candidates_test:
        raise FileNotFoundError(
            f"Could not find train.csv/test.csv under /kaggle/input. "
            f"Checked defaults train_path={train_path}, test_path={test_path}."
        )

    # Prefer paths that include 'playground' or 's5e12' if present
    def _score(p: Path) -> int:
        s = str(p).lower()
        score = 0
        if "playground" in s:
            score += 2
        if "s5e12" in s or "s5-e12" in s:
            score += 2
        return score

    best_train = max(candidates_train, key=_score)
    best_test = max(candidates_test, key=_score)

    return str(best_train), str(best_test)

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
    border_count = params.get("border_count", 128)

    model_kwargs = dict(
        task_type="GPU",
        devices="0",
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
    train_path, test_path = resolve_kaggle_paths(TRAIN_PATH, TEST_PATH)
    df_train_total, df_test = load_data(train_path, test_path)
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
