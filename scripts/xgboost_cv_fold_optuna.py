# %%
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import xgboost as xgb
from xgboost import XGBClassifier


# =============================================================================
# PATHS (Kaggle-ready)
# =============================================================================
# Kaggle inputs are mounted under /kaggle/input, outputs should go to /kaggle/working.
# If you don't know the exact dataset folder name, we auto-detect train.csv/test.csv under /kaggle/input.
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
OUTPUT_DIR = "/kaggle/working"  # Kaggle
OUTPUT_PREFIX = "xgb_optuna"

# Uncomment for local runs
# TRAIN_PATH = "../data/raw/playground-series-s5e12/train.csv"
# TEST_PATH = "../data/raw/playground-series-s5e12/test.csv"
# OUTPUT_DIR = "../submissions"
# =============================================================================
# Helper for XGBoost early stopping (version-agnostic)
# =============================================================================
def _fit_model_with_es(
    model: XGBClassifier,
    X_tr: np.ndarray,
    y_tr: pd.Series,
    X_va: np.ndarray,
    y_va: pd.Series,
) -> None:
    """
    XGBoost sklearn API differs across versions:
      - some support `callbacks=...`
      - some support `early_stopping_rounds=...` in fit()
      - some support `early_stopping_rounds` as a constructor param
    We try these in order and fall back to no-ES if unavailable.
    """
    # 1) Newer API: callbacks
    try:
        es_cb = xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            callbacks=[es_cb],
        )
        return
    except TypeError:
        pass

    # 2) Older sklearn API: early_stopping_rounds in fit()
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        return
    except TypeError:
        pass

    # 3) Some versions accept early_stopping_rounds as an init/set_params param
    try:
        model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        return
    except TypeError:
        pass

    # 4) Last resort: train without early stopping
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )


def resolve_kaggle_paths(train_path: str, test_path: str) -> tuple[str, str]:
    """
    Resolve train/test CSV paths by:
      1) Using provided paths if they exist
      2) Otherwise searching /kaggle/input/**/train.csv and test.csv
    """
    train_p = Path(train_path)
    test_p = Path(test_path)

    if train_p.exists() and test_p.exists():
        return str(train_p), str(test_p)

    base = Path("/kaggle/input")
    if not base.exists():
        # not on Kaggle; just return original paths (will raise later if missing)
        return train_path, test_path

    candidates_train = sorted(base.rglob("train.csv"))
    candidates_test = sorted(base.rglob("test.csv"))

    if not candidates_train or not candidates_test:
        raise FileNotFoundError(
            f"Could not find train.csv/test.csv under /kaggle/input. "
            f"Checked defaults train_path={train_path}, test_path={test_path}."
        )

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


# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
TARGET = "diagnosed_diabetes"
ID_COLUMN = "id"

# final CV folds is irrelevant here (this is tuning-only),
# but keep in case you want to re-use the file later.
N_FOLDS = 5

# categoricals for manual encoding (XGBoost can't do raw strings)
ORDINAL_COLS = ["education_level", "income_level", "smoking_status"]
ORDINAL_CATEGORIES = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"],
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    ["Current", "Former", "Never"],
]
OHE_COLS = ["gender", "ethnicity", "employment_status"]

# Optuna tuning
RUN_OPTUNA = True
OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT_SEC = None
OPTUNA_TUNE_FOLDS = 3  # fewer folds during tuning for speed
OPTUNA_DIRECTION = "maximize"
OPTUNA_SHOW_PROGRESS = True

# training control (kept quiet)
EARLY_STOPPING_ROUNDS = 100
EVAL_VERBOSE = 0  # 0 => quiet

# XGBoost compute
USE_GPU = True  # Kaggle GPU: set True; local CPU: set False
TREE_METHOD_CPU = "hist"
TREE_METHOD_GPU = "hist"  # in xgboost>=2, use device="cuda" with hist


# =============================================================================
# DATA
# =============================================================================
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df = df.copy()
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["non_hdl"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["ldl_hdl"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + eps)
    df["tg_hdl"] = df["triglycerides"] / (df["hdl_cholesterol"] + eps)
    df["total_hdl"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + eps)
    df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + eps)
    return df


def split_features_target(df: pd.DataFrame, target: str, id_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target, id_column]).copy()
    y = df[target].astype(int)
    return X, y


def encode_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    ordinal_cols: list[str],
    ordinal_categories: list[list[str]],
    ohe_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fit encoders on train fold only. Return numpy arrays + aligned feature names.
    (No test encoding here; tuning only needs train/val.)
    """
    # ordinal
    ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
    X_train.loc[:, ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])
    X_val.loc[:, ordinal_cols] = ordinal_encoder.transform(X_val[ordinal_cols])

    # one-hot
    ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_ohe = ohe_encoder.fit_transform(X_train[ohe_cols])
    X_val_ohe = ohe_encoder.transform(X_val[ohe_cols])

    base_feature_names = X_train.drop(columns=ohe_cols).columns.tolist()
    ohe_feature_names = ohe_encoder.get_feature_names_out(ohe_cols).tolist()
    feature_names = base_feature_names + ohe_feature_names

    X_train_arr = np.concatenate([X_train.drop(columns=ohe_cols).to_numpy(), X_train_ohe], axis=1)
    X_val_arr = np.concatenate([X_val.drop(columns=ohe_cols).to_numpy(), X_val_ohe], axis=1)
    return X_train_arr, X_val_arr, feature_names


# =============================================================================
# MODEL + CV
# =============================================================================
@dataclass
class CVScore:
    mean_auc: float
    fold_aucs: list[float]


def build_model(seed: int, params: dict) -> XGBClassifier:
    """
    XGBoost sklearn wrapper. Early stopping handled via callback (version-safe).
    """
    device = "cuda" if USE_GPU else "cpu"
    tree_method = TREE_METHOD_GPU if USE_GPU else TREE_METHOD_CPU

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        gamma=float(params["gamma"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        tree_method=tree_method,
        device=device,
        n_jobs=-1,
    )


def cv_score_auc(df_train_total: pd.DataFrame, seed: int, n_folds: int, params: dict) -> CVScore:
    X_total, y_total = split_features_target(df_train_total, TARGET, ID_COLUMN)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs: list[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_total, y_total), start=1):
        X_tr = X_total.iloc[tr_idx].copy()
        y_tr = y_total.iloc[tr_idx]
        X_va = X_total.iloc[va_idx].copy()
        y_va = y_total.iloc[va_idx]

        X_tr_arr, X_va_arr, _ = encode_fold(X_tr, X_va, ORDINAL_COLS, ORDINAL_CATEGORIES, OHE_COLS)

        model = build_model(seed, params)

        # Use version-agnostic early stopping helper
        _fit_model_with_es(model, X_tr_arr, y_tr, X_va_arr, y_va)

        y_va_pred = model.predict_proba(X_va_arr)[:, 1]
        auc = roc_auc_score(y_va, y_va_pred)
        fold_aucs.append(float(auc))

    return CVScore(mean_auc=float(np.mean(fold_aucs)), fold_aucs=fold_aucs)


# =============================================================================
# OPTUNA
# =============================================================================
def optuna_objective(trial: optuna.Trial, df_train_total: pd.DataFrame) -> float:
    """
    Tight-ish search space for tabular XGBoost.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }

    score = cv_score_auc(
        df_train_total=df_train_total,
        seed=SEED,
        n_folds=OPTUNA_TUNE_FOLDS,
        params=params,
    )
    return score.mean_auc


def run_optuna(df_train_total: pd.DataFrame) -> tuple[dict, float, optuna.Study]:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction=OPTUNA_DIRECTION, sampler=sampler)

    study.optimize(
        lambda t: optuna_objective(t, df_train_total),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SEC,
        show_progress_bar=OPTUNA_SHOW_PROGRESS,
    )

    return study.best_params, float(study.best_value), study


def save_optuna_outputs(best_params: dict, best_value: float, study: optuna.Study, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_path = out_dir / f"{OUTPUT_PREFIX}_{ts}_best.json"
    with best_path.open("w") as f:
        json.dump({"best_value_auc": best_value, "best_params": best_params}, f, indent=2)
    print(f"Saved best params: {best_path}")

    trials_path = out_dir / f"{OUTPUT_PREFIX}_{ts}_trials.csv"
    df_trials = study.trials_dataframe()
    df_trials.to_csv(trials_path, index=False)
    print(f"Saved trials: {trials_path}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    train_path, test_path = resolve_kaggle_paths(TRAIN_PATH, TEST_PATH)
    df_train_total, _ = load_data(train_path, test_path)

    df_train_total = feature_engineering(df_train_total)

    if RUN_OPTUNA:
        best_params, best_value, study = run_optuna(df_train_total)
        print("Optuna best AUC:", best_value)
        print("Optuna best params:", best_params)
        save_optuna_outputs(best_params, best_value, study, OUTPUT_DIR)
    else:
        # quick sanity run with defaults (not recommended for tuning, but useful for debugging)
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 4,
            "min_child_weight": 3.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        score = cv_score_auc(df_train_total, SEED, OPTUNA_TUNE_FOLDS, default_params)
        print("Default mean AUC:", score.mean_auc, "| folds:", score.fold_aucs)


if __name__ == "__main__":
    main()
# %%
