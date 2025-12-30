# %%
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

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

CATBOOST_PARAMS_PATH = "../parameters/catboost_params.json"

CATBOOST_REQUIRED_KEYS = [
    "CATBOOST_NUM_ITERATIONS",
    "CATBOOST_LEARNING_RATE",
    "CATBOOST_DEPTH",
    "CATBOOST_EARLY_STOPPING_ROUNDS",
    "CATBOOST_EVAL_METRIC",
    "CATBOOST_LOSS_FUNCTION",
    "CATBOOST_L2_LEAF_REG",
    "CATBOOST_RANDOM_STRENGTH",
    "CATBOOST_BAGGING_TEMPERATURE",
    "CATBOOST_BORDER_COUNT",
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
def load_catboost_params(path):
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    missing = [k for k in CATBOOST_REQUIRED_KEYS if k not in params]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing keys in {path}: {missing_str}")
    return params

# %%
def build_model(seed, params):
    return CatBoostClassifier(
        iterations=params["CATBOOST_NUM_ITERATIONS"],
        learning_rate=params["CATBOOST_LEARNING_RATE"],
        depth=params["CATBOOST_DEPTH"],
        random_seed=seed,
        early_stopping_rounds=params["CATBOOST_EARLY_STOPPING_ROUNDS"],
        eval_metric=params["CATBOOST_EVAL_METRIC"],
        verbose=True,
        loss_function=params["CATBOOST_LOSS_FUNCTION"],
        l2_leaf_reg=params["CATBOOST_L2_LEAF_REG"],
        allow_writing_files=False,
        random_strength=params["CATBOOST_RANDOM_STRENGTH"],
        bagging_temperature=params["CATBOOST_BAGGING_TEMPERATURE"],
        border_count=params["CATBOOST_BORDER_COUNT"],
    )

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

        model = build_model(seed, model_params)
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

    model_params = load_catboost_params(CATBOOST_PARAMS_PATH)
    cv_results = train_cv(
        df_train_total,
        df_test,
        SEED,
        N_FOLDS,
        TARGET,
        ID_COLUMN,
        EVAL_VERBOSE,
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
