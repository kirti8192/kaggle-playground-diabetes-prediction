
# %%
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb

import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
TARGET = "diagnosed_diabetes"
ID_COLUMN = "id"
N_FOLDS = 5

TRAIN_PATH = "../data/raw/playground-series-s5e12/train.csv"
TEST_PATH = "../data/raw/playground-series-s5e12/test.csv"

SUBMISSION_DIR = "../submissions"
SUBMISSION_PREFIX = "submission"
SUBMISSION_TAG = "catboost_lgbm_ensemble"

# toggles
PLOT_AUC_CURVES = True
PLOT_FEATURE_IMPORTANCE = True
SAVE_SUBMISSION = True

EVAL_VERBOSE = 10
TOP_N_IMPORTANCE = 30

# categorical columns (handled natively by CatBoost / LightGBM)
CAT_COLS = [
    "education_level",
    "income_level",
    "smoking_status",
    "gender",
    "ethnicity",
    "employment_status",
]

# --- CatBoost: Optuna-optimized hyperparams (retain as-is) ---
CATBOOST_NUM_ITERATIONS = 3838
CATBOOST_LEARNING_RATE = 0.05359374417837557
CATBOOST_DEPTH = 5
CATBOOST_EARLY_STOPPING_ROUNDS = 100
CATBOOST_EVAL_METRIC = "AUC"
CATBOOST_LOSS_FUNCTION = "Logloss"
CATBOOST_L2_LEAF_REG = 5.104333350702094
CATBOOST_RANDOM_STRENGTH = 1.747464298900436
CATBOOST_BAGGING_TEMPERATURE = 0.5210398649951489
CATBOOST_BORDER_COUNT = 254

# --- LightGBM: treat these as optimal for now (will paste Optuna params later) ---
LGBM_N_ESTIMATORS = 4890
LGBM_LEARNING_RATE = 0.12511547004171678
LGBM_NUM_LEAVES = 24
LGBM_MAX_DEPTH = 2
LGBM_MIN_CHILD_SAMPLES = 134
LGBM_SUBSAMPLE = 0.9624457541570358
LGBM_COLSAMPLE_BYTREE = 0.9506589893038976
LGBM_REG_ALPHA = 2.51545560767482
LGBM_REG_LAMBDA = 4.83334465226836

LGBM_EARLY_STOPPING_ROUNDS = 200
LGBM_VERBOSE = 100

# LightGBM GPU (optional): set to "gpu" only if your LightGBM build supports it
LGBM_DEVICE_TYPE = "cpu"  # "cpu" or "gpu"


# =============================================================================
# IO + FEATURES
# =============================================================================
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple ratio/interaction features (same for both models).
    """
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


# =============================================================================
# MODELS
# =============================================================================
def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=CATBOOST_NUM_ITERATIONS,
        learning_rate=CATBOOST_LEARNING_RATE,
        depth=CATBOOST_DEPTH,
        random_seed=seed,
        early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
        eval_metric=CATBOOST_EVAL_METRIC,
        loss_function=CATBOOST_LOSS_FUNCTION,
        l2_leaf_reg=CATBOOST_L2_LEAF_REG,
        random_strength=CATBOOST_RANDOM_STRENGTH,
        bagging_temperature=CATBOOST_BAGGING_TEMPERATURE,
        border_count=CATBOOST_BORDER_COUNT,
        allow_writing_files=False,
        verbose=False,
    )


def build_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        num_leaves=LGBM_NUM_LEAVES,
        max_depth=LGBM_MAX_DEPTH,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        subsample=LGBM_SUBSAMPLE,
        colsample_bytree=LGBM_COLSAMPLE_BYTREE,
        reg_alpha=LGBM_REG_ALPHA,
        reg_lambda=LGBM_REG_LAMBDA,
        random_state=seed,
        n_jobs=-1,
        device_type=LGBM_DEVICE_TYPE,
    )


# =============================================================================
# CV TRAINING (OOF + TEST PREDS)
# =============================================================================
@dataclass
class CVResult:
    name: str
    oof_pred: np.ndarray
    test_pred: np.ndarray
    fold_scores: list[float]
    fold_eval_results: list[dict]
    models: list
    feature_names: list[str]


def train_cv_catboost(
    df_train_total: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int,
    n_folds: int,
    target: str,
    id_column: str,
    eval_verbose: int,
) -> CVResult:
    X_total, y_total = split_features_target(df_train_total, target, id_column)
    X_test = df_test.drop(columns=[id_column]).copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(X_total), dtype=float)
    test_preds = []
    fold_scores: list[float] = []
    fold_eval_results: list[dict] = []
    models = []

    feature_names = X_total.columns.tolist()
    cat_feature_indices = [X_total.columns.get_loc(col) for col in CAT_COLS if col in X_total.columns]

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_total, y_total), start=1):
        X_train, y_train = X_total.iloc[tr_idx], y_total.iloc[tr_idx]
        X_val, y_val = X_total.iloc[va_idx], y_total.iloc[va_idx]

        model = build_catboost(seed)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=eval_verbose,
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[va_idx] = val_pred
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(float(fold_auc))
        print(f"[CatBoost] Fold {fold_idx} AUC: {fold_auc:.5f}")

        fold_eval_results.append(model.get_evals_result())
        models.append(model)
        test_preds.append(model.predict_proba(X_test)[:, 1])

    test_pred = np.mean(np.vstack(test_preds), axis=0)
    return CVResult(
        name="catboost",
        oof_pred=oof,
        test_pred=test_pred,
        fold_scores=fold_scores,
        fold_eval_results=fold_eval_results,
        models=models,
        feature_names=feature_names,
    )


def train_cv_lgbm(
    df_train_total: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int,
    n_folds: int,
    target: str,
    id_column: str,
    eval_verbose: int,
) -> CVResult:
    X_total, y_total = split_features_target(df_train_total, target, id_column)
    X_test = df_test.drop(columns=[id_column]).copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(X_total), dtype=float)
    test_preds = []
    fold_scores: list[float] = []
    fold_eval_results: list[dict] = []
    models = []

    feature_names = X_total.columns.tolist()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_total, y_total), start=1):
        X_train, y_train = X_total.iloc[tr_idx].copy(), y_total.iloc[tr_idx]
        X_val, y_val = X_total.iloc[va_idx].copy(), y_total.iloc[va_idx]
        X_test_fold = X_test.copy()

        # LightGBM native categoricals: dtype=category
        for c in CAT_COLS:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype("category")
                X_val[c] = X_val[c].astype("category")
                X_test_fold[c] = X_test_fold[c].astype("category")

        model = build_lgbm(seed)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc", "binary_logloss"],
            categorical_feature=[c for c in CAT_COLS if c in X_train.columns],
            callbacks=[
                lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=eval_verbose),
            ],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[va_idx] = val_pred
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(float(fold_auc))
        print(f"[LightGBM] Fold {fold_idx} AUC: {fold_auc:.5f}")

        fold_eval_results.append(model.evals_result_)
        models.append(model)
        test_preds.append(model.predict_proba(X_test_fold)[:, 1])

    test_pred = np.mean(np.vstack(test_preds), axis=0)
    return CVResult(
        name="lightgbm",
        oof_pred=oof,
        test_pred=test_pred,
        fold_scores=fold_scores,
        fold_eval_results=fold_eval_results,
        models=models,
        feature_names=feature_names,
    )


# =============================================================================
# ENSEMBLE (OOF weight search)
# =============================================================================
def find_best_blend_weight(y_true: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray) -> tuple[float, float]:
    """
    Search w in [0,1] for blend = w*a + (1-w)*b that maximizes AUC on OOF.
    Returns (best_w_for_a, best_auc).
    """
    best_w = 0.5
    best_auc = -1.0
    for w in np.linspace(0.0, 1.0, 101):
        blended = w * oof_a + (1.0 - w) * oof_b
        auc = roc_auc_score(y_true, blended)
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)
    return best_w, float(best_auc)



# =============================================================================
# PLOTTING
# =============================================================================

# --- Helper functions for extracting AUC curves ---
def _extract_catboost_auc_curve(evals_result: dict) -> list[float] | None:
    # CatBoost eval results can be keyed by "validation" or "validation_0"
    val = evals_result.get("validation") or evals_result.get("validation_0") or {}
    for k in ("AUC", "auc"):
        if k in val:
            return list(val[k])
    return None


def _extract_lgbm_auc_curve(evals_result: dict) -> list[float] | None:
    # LightGBM evals_result_ stores validation under "valid_0"
    val = evals_result.get("valid_0", {})
    if "auc" in val:
        return list(val["auc"])
    return None


def plot_auc_curves_combined(cb_eval_results: list[dict], lgbm_eval_results: list[dict]) -> None:
    """
    Plot validation AUC curves for CatBoost and LightGBM on the SAME figure.
    One line per fold per model:
      - CatBoost folds: solid lines
      - LightGBM folds: dashed lines
    """
    plt.figure(figsize=(10, 6))

    n_folds = max(len(cb_eval_results), len(lgbm_eval_results))
    for i in range(n_folds):
        if i < len(cb_eval_results):
            cb_curve = _extract_catboost_auc_curve(cb_eval_results[i])
            if cb_curve:
                plt.plot(
                    range(1, len(cb_curve) + 1),
                    cb_curve,
                    linestyle="-",
                    label=f"CatBoost Fold {i+1}",
                )

        if i < len(lgbm_eval_results):
            lgb_curve = _extract_lgbm_auc_curve(lgbm_eval_results[i])
            if lgb_curve:
                plt.plot(
                    range(1, len(lgb_curve) + 1),
                    lgb_curve,
                    linestyle="--",
                    label=f"LightGBM Fold {i+1}",
                )

    plt.title("Validation AUC per Iteration — CatBoost vs LightGBM (all folds)")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_feature_importance_lgbm(model: LGBMClassifier, feature_names: list[str], top_n: int) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    if top_n:
        idx = idx[: min(top_n, len(idx))]
    idx_plot = idx[::-1]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(idx_plot)), importances[idx_plot], align="center")
    plt.yticks(range(len(idx_plot)), [feature_names[i] for i in idx_plot])
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.show()


def plot_feature_importance_catboost(model: CatBoostClassifier, feature_names: list[str], top_n: int) -> None:
    importances = model.get_feature_importance()
    idx = np.argsort(importances)[::-1]
    if top_n:
        idx = idx[: min(top_n, len(idx))]
    idx_plot = idx[::-1]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(idx_plot)), importances[idx_plot], align="center")
    plt.yticks(range(len(idx_plot)), [feature_names[i] for i in idx_plot])
    plt.title("CatBoost Feature Importance")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SUBMISSION
# =============================================================================
def save_submission(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    id_column: str,
    target: str,
    submission_dir: str,
    submission_prefix: str,
    submission_tag: str,
) -> Path:
    out_dir = Path(submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{submission_tag}" if submission_tag else ""
    out_path = out_dir / f"{submission_prefix}_{timestamp}{suffix}.csv"

    pd.DataFrame({id_column: df_test[id_column], target: y_pred}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    df_train_total, df_test = load_data(TRAIN_PATH, TEST_PATH)

    df_train_total = feature_engineering(df_train_total)
    df_test = feature_engineering(df_test)

    y_true = df_train_total[TARGET].astype(int).to_numpy()

    # Train both models with CV
    cb_res = train_cv_catboost(df_train_total, df_test, SEED, N_FOLDS, TARGET, ID_COLUMN, EVAL_VERBOSE)
    lgbm_res = train_cv_lgbm(df_train_total, df_test, SEED, N_FOLDS, TARGET, ID_COLUMN, LGBM_VERBOSE)

    print(f"\nCatBoost mean AUC: {np.mean(cb_res.fold_scores):.5f} ± {np.std(cb_res.fold_scores):.5f}")
    print(f"LightGBM mean AUC: {np.mean(lgbm_res.fold_scores):.5f} ± {np.std(lgbm_res.fold_scores):.5f}")

    # Find best blend weight on OOF predictions
    w_cb, best_oof_auc = find_best_blend_weight(y_true, cb_res.oof_pred, lgbm_res.oof_pred)
    print(f"\nBest OOF blend: w_catboost={w_cb:.2f}, w_lgbm={1-w_cb:.2f} | OOF AUC={best_oof_auc:.5f}")

    # Blend test predictions with the same weight
    y_test_pred = w_cb * cb_res.test_pred + (1.0 - w_cb) * lgbm_res.test_pred

    # Optional plots
    if PLOT_AUC_CURVES:
        plot_auc_curves_combined(cb_res.fold_eval_results, lgbm_res.fold_eval_results)

    if PLOT_FEATURE_IMPORTANCE:
        plot_feature_importance_catboost(cb_res.models[-1], cb_res.feature_names, TOP_N_IMPORTANCE)
        plot_feature_importance_lgbm(lgbm_res.models[-1], lgbm_res.feature_names, TOP_N_IMPORTANCE)

    # Save submission
    if SAVE_SUBMISSION:
        save_submission(
            df_test=df_test,
            y_pred=y_test_pred,
            id_column=ID_COLUMN,
            target=TARGET,
            submission_dir=SUBMISSION_DIR,
            submission_prefix=SUBMISSION_PREFIX,
            submission_tag=SUBMISSION_TAG,
        )


if __name__ == "__main__":
    main()
# %%
