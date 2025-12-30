# %%
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

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
SUBMISSION_TAG = "xgboost_final"

PLOT_AUC_CURVES = True
PLOT_FEATURE_IMPORTANCE = True
SAVE_SUBMISSION = True
EVAL_VERBOSE = 100

SAVE_LOGS = True
LOG_DIR = "../logs"
LOG_BASENAME = "cv_fold_xgboost"

SAVE_MODEL = True
MODEL_DIR = "../models"
MODEL_FILENAME = "xgboost_optuna_model.json"

ORDINAL_COLS = [
    "education_level",
    "income_level",
    "smoking_status",
]
ORDINAL_CATEGORIES = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"],
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    ["Current", "Former", "Never"],
]
OHE_COLS = [
    "gender",
    "ethnicity",
    "employment_status",
]

XGB_PARAMS_PATH = "../parameters/xgboost_optuna_params.json"

XGB_OBJECTIVE = "binary:logistic"
XGB_EVAL_METRIC = ["logloss", "auc"]
XGB_TREE_METHOD = "hist"
XGB_EARLY_STOPPING_ROUNDS = 100

XGB_REQUIRED_KEYS = [
    "XGB_N_ESTIMATORS",
    "XGB_LEARNING_RATE",
    "XGB_MAX_DEPTH",
    "XGB_MIN_CHILD_WEIGHT",
    "XGB_SUBSAMPLE",
    "XGB_COLSAMPLE_BYTREE",
    "XGB_GAMMA",
    "XGB_REG_ALPHA",
    "XGB_REG_LAMBDA",
]

# %%
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# %%
def feature_engineering(df):
    """
    easily available medical metrics that captures non-linearity
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

# %%
def split_features_target(df, target, id_column):
    X = df.drop(columns=[target, id_column]).copy()
    y = df[target].astype(int)
    return X, y

# %%
def encode_fold(X_train, X_val, X_test, ordinal_cols, ordinal_categories, ohe_cols):
    """
    Perform encoding of categorical features
    Ordinal for features with a natural ordering
    OneHot for features with no natural ordering
    """

    ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
    X_train.loc[:, ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])
    X_val.loc[:, ordinal_cols] = ordinal_encoder.transform(X_val[ordinal_cols])
    X_test.loc[:, ordinal_cols] = ordinal_encoder.transform(X_test[ordinal_cols])

    ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_ohe = ohe_encoder.fit_transform(X_train[ohe_cols])
    X_val_ohe = ohe_encoder.transform(X_val[ohe_cols])
    X_test_ohe = ohe_encoder.transform(X_test[ohe_cols])

    base_feature_names = X_train.drop(columns=ohe_cols).columns.tolist()
    ohe_feature_names = ohe_encoder.get_feature_names_out(ohe_cols).tolist()
    feature_names = base_feature_names + ohe_feature_names

    X_train_arr = np.concatenate([X_train.drop(columns=ohe_cols).to_numpy(), X_train_ohe], axis=1)
    X_val_arr = np.concatenate([X_val.drop(columns=ohe_cols).to_numpy(), X_val_ohe], axis=1)
    X_test_arr = np.concatenate([X_test.drop(columns=ohe_cols).to_numpy(), X_test_ohe], axis=1)

    return X_train_arr, X_val_arr, X_test_arr, feature_names

# %%
def load_xgb_params(path):
    """
    Load parameters obtained with optuna
    """

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    missing = [k for k in XGB_REQUIRED_KEYS if k not in params]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing keys in {path}: {missing_str}")
    return params

# %%
def build_model(seed, params):

    """
    Build the model
    """

    return XGBClassifier(
        objective=XGB_OBJECTIVE,
        eval_metric=XGB_EVAL_METRIC,
        random_state=seed,
        max_depth=params["XGB_MAX_DEPTH"],
        min_child_weight=params["XGB_MIN_CHILD_WEIGHT"],
        gamma=params["XGB_GAMMA"],
        n_estimators=params["XGB_N_ESTIMATORS"],
        learning_rate=params["XGB_LEARNING_RATE"],
        subsample=params["XGB_SUBSAMPLE"],
        colsample_bytree=params["XGB_COLSAMPLE_BYTREE"],
        reg_lambda=params["XGB_REG_LAMBDA"],
        reg_alpha=params["XGB_REG_ALPHA"],
        tree_method=XGB_TREE_METHOD,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
    )

# %%
def collect_xgb_logs(evals_result, fold_idx):

    """
    Collect the logs from the model, used to plot logloss and auc curves later
    """

    records = []
    split_map = {
        "validation_0": "train",
        "validation_1": "val",
    }
    metric_map = {
        "logloss": "logloss",
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
                        "model": "xgboost",
                        "fold": fold_idx,
                        "iteration": iteration,
                        "split": split_label,
                        "metric": metric_label,
                        "value": value,
                    }
                )
    return records

# %%
def get_xgb_best_iteration(model, evals_result):

    """
    store best iteration from model for later plotting and evaluation
    """

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is not None and best_iteration >= 0:
        return int(best_iteration) + 1
    val_metrics = evals_result.get("validation_1", {})
    values = val_metrics.get("auc") or val_metrics.get("logloss")
    if values:
        return len(values)
    return None

# %%
def get_xgb_best_metrics(evals_result, best_iteration):

    """
    store stats corresponding to best iteration from model for later plotting and evaluation
    """

    if best_iteration is None:
        return None, None
    idx = best_iteration - 1
    val_metrics = evals_result.get("validation_1", {})
    auc_values = val_metrics.get("auc")
    logloss_values = val_metrics.get("logloss")
    best_val_auc = auc_values[idx] if auc_values and idx < len(auc_values) else None
    best_val_logloss = logloss_values[idx] if logloss_values and idx < len(logloss_values) else None
    return best_val_auc, best_val_logloss

# %%
def train_cv(df_train_total, df_test, seed, n_folds, target, id_column, eval_verbose, model_params):

    """ 
    Main training loop
    """

    # split data into features and target
    X_total, y_total = split_features_target(df_train_total, target, id_column)

    # stratified k-fold cross validation (because target is imbalanced)
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    # initialize variables
    fold_eval_results = []
    models = []
    feature_names_per_fold = []
    test_pred_folds = []
    val_scores = []
    log_records = []
    best_records = []

    # iterate over folds
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_total, y_total), start=1):

        # get train and validation data ready 
        df_train = df_train_total.iloc[train_index]
        df_val = df_train_total.iloc[val_index]

        X_train = df_train.drop(columns=[target, id_column]).copy()
        y_train = df_train[target].astype(int)

        X_val = df_val.drop(columns=[target, id_column]).copy()
        y_val = df_val[target].astype(int)

        X_test = df_test.drop(columns=[id_column]).copy()

        X_train_arr, X_val_arr, X_test_arr, feature_names = encode_fold(
            X_train, X_val, X_test, ORDINAL_COLS, ORDINAL_CATEGORIES, OHE_COLS
        )

        # train model
        model = build_model(seed, model_params)
        model.fit(
            X_train_arr,
            y_train,
            eval_set=[(X_train_arr, y_train), (X_val_arr, y_val)],  # validation on train data so that we can plot logloss and auc curves
            verbose=eval_verbose,
        )

        # predict validation data
        y_val_pred = model.predict_proba(X_val_arr)[:, 1]

        # calculate validation auc
        val_auc = roc_auc_score(y_val, y_val_pred)
        print(f"Fold {fold_idx} Validation AUC: {val_auc:.4f}")

        # store evaluation results
        evals_result = model.evals_result()
        fold_eval_results.append(evals_result)
        log_records.extend(collect_xgb_logs(evals_result, fold_idx))
        best_iteration = get_xgb_best_iteration(model, evals_result)
        best_val_auc, best_val_logloss = get_xgb_best_metrics(evals_result, best_iteration)
        if best_iteration is not None:
            best_records.append(
                {
                    "model": "xgboost",
                    "fold": fold_idx,
                    "best_iteration": best_iteration,
                    "best_val_auc": best_val_auc,
                    "best_val_logloss": best_val_logloss,
                }
            )

        # store model
        models.append(model)

        feature_names_per_fold.append(feature_names)
        test_pred_folds.append(model.predict_proba(X_test_arr)[:, 1])
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
    Plot validation AUC curves
    """

    plt.figure(figsize=(10, 6))
    for fold_idx, evals_result in enumerate(fold_eval_results, start=1):
        val_auc_curve = evals_result["validation_0"]["auc"]
        plt.plot(range(1, len(val_auc_curve) + 1), val_auc_curve, label=f"Fold {fold_idx}")
    plt.xlabel("Boosting Round")
    plt.ylabel("Validation AUC")
    plt.title("Validation AUC Curves per Fold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
def plot_feature_importance(model, feature_names, top_n):
    """
    Plot feature importance
    """

    feature_importances = model.feature_importances_
    if len(feature_importances) != len(feature_names):
        raise ValueError(
            f"Mismatch: importances={len(feature_importances)} vs names={len(feature_names)}. "
            "This indicates a preprocessing/feature-name alignment issue."
        )

    sorted_idx = np.argsort(feature_importances)
    if top_n is not None and top_n > 0:
        top_n = min(top_n, len(sorted_idx))
        sorted_idx = sorted_idx[-top_n:]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()

# %%
def save_submission(df_test, y_pred, id_column, target, submission_dir, submission_prefix, submission_tag):

    """
    Save submission csv file
    """

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
    """
    Save model
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    booster = model.get_booster()
    booster.save_model(str(path))
    print(f"Saved: {path}")


# %%
def save_logs(log_records, best_records, log_dir, log_basename):

    """
    Save logs for later analysis
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_records:
        log_df = pd.DataFrame(log_records)
        log_df = log_df[["model", "fold", "iteration", "split", "metric", "value"]]
        log_df.to_csv(log_dir / f"{log_basename}_stats.csv", index=False)
    if best_records:
        best_df = pd.DataFrame(best_records)
        best_df = best_df[["model", "fold", "best_iteration", "best_val_auc", "best_val_logloss"]]
        best_df.to_csv(log_dir / f"{log_basename}_summary_stats.csv", index=False)

# %%

df_train_total, df_test = load_data(TRAIN_PATH, TEST_PATH)
df_train_total = feature_engineering(df_train_total)
df_test = feature_engineering(df_test)

model_params = load_xgb_params(XGB_PARAMS_PATH)
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

if SAVE_LOGS:
    save_logs(
        cv_results["log_records"],
        cv_results["best_records"],
        LOG_DIR,
        LOG_BASENAME,
    )
# %%
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
# %%
