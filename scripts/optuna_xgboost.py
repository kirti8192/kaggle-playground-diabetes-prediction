# %%
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import optuna
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

TUNE_FOLDS = 3
OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT_SEC = None
EARLY_STOPPING_ROUNDS = 100

KAGGLE_TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
KAGGLE_TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
LOCAL_TRAIN_PATH = "../data/raw/playground-series-s5e12/train.csv"
LOCAL_TEST_PATH = "../data/raw/playground-series-s5e12/test.csv"

PARAMS_DIR = "../parameters"
PARAMS_FILENAME = "xgboost_optuna_params.json"

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

# %%

def resolve_kaggle_paths(train_path, test_path, local_train_path, local_test_path):
    if Path(train_path).exists() and Path(test_path).exists():
        return train_path, test_path

    if Path("/kaggle/input").exists():
        candidates_train = sorted(Path("/kaggle/input").rglob("train.csv"))
        candidates_test = sorted(Path("/kaggle/input").rglob("test.csv"))
        if candidates_train and candidates_test:
            return str(candidates_train[0]), str(candidates_test[0])

    if Path(local_train_path).exists() and Path(local_test_path).exists():
        return local_train_path, local_test_path

    raise FileNotFoundError("train.csv/test.csv not found for Kaggle or local paths")


# %%
def load_data(train_path):
    train_df = pd.read_csv(train_path)
    return train_df


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
def encode_fold(X_train, X_val, ordinal_cols, ordinal_categories, ohe_cols):
    """
    Perform encoding of categorical features
    Ordinal for features with a natural ordering
    OneHot for features with no natural ordering
    """
    ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
    X_train.loc[:, ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])
    X_val.loc[:, ordinal_cols] = ordinal_encoder.transform(X_val[ordinal_cols])

    ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_ohe = ohe_encoder.fit_transform(X_train[ohe_cols])
    X_val_ohe = ohe_encoder.transform(X_val[ohe_cols])

    X_train_arr = np.concatenate([X_train.drop(columns=ohe_cols).to_numpy(), X_train_ohe], axis=1)
    X_val_arr = np.concatenate([X_val.drop(columns=ohe_cols).to_numpy(), X_val_ohe], axis=1)

    return X_train_arr, X_val_arr


# %%
def build_model(seed, params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=seed,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


# %%
def tune_xgboost(df_train_total, splits):
    X_total, y_total = split_features_target(df_train_total, TARGET, ID_COLUMN)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 800, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }

        fold_scores = []
        for fold_idx, (train_index, val_index) in enumerate(splits, start=1):
            df_train = df_train_total.iloc[train_index]
            df_val = df_train_total.iloc[val_index]

            X_train = df_train.drop(columns=[TARGET, ID_COLUMN]).copy()
            y_train = df_train[TARGET].astype(int)
            X_val = df_val.drop(columns=[TARGET, ID_COLUMN]).copy()
            y_val = df_val[TARGET].astype(int)

            X_train_arr, X_val_arr = encode_fold(
                X_train,
                X_val,
                ORDINAL_COLS,
                ORDINAL_CATEGORIES,
                OHE_COLS,
            )

            model = build_model(SEED, params)
            model.fit(
                X_train_arr,
                y_train,
                eval_set=[(X_val_arr, y_val)],
                verbose=False,
            )

            y_val_pred = model.predict_proba(X_val_arr)[:, 1]
            auc = roc_auc_score(y_val, y_val_pred)
            fold_scores.append(auc)

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT_SEC)
    return study


# %%
def save_best_params(study, output_dir, filename):
    output_dir = Path(output_dir)
    if Path("/kaggle/working").exists():
        output_dir = Path("/kaggle/working") / "parameters"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_params = study.best_trial.params
    payload = {
        "XGB_N_ESTIMATORS": int(best_params["n_estimators"]),
        "XGB_LEARNING_RATE": best_params["learning_rate"],
        "XGB_MAX_DEPTH": int(best_params["max_depth"]),
        "XGB_MIN_CHILD_WEIGHT": best_params["min_child_weight"],
        "XGB_SUBSAMPLE": best_params["subsample"],
        "XGB_COLSAMPLE_BYTREE": best_params["colsample_bytree"],
        "XGB_GAMMA": best_params["gamma"],
        "XGB_REG_ALPHA": best_params["reg_alpha"],
        "XGB_REG_LAMBDA": best_params["reg_lambda"],
        "best_value_auc": study.best_value,
        "timestamp": datetime.now().isoformat(),
    }
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


# %%
train_path, _ = resolve_kaggle_paths(
    KAGGLE_TRAIN_PATH,
    KAGGLE_TEST_PATH,
    LOCAL_TRAIN_PATH,
    LOCAL_TEST_PATH,
)

train_df = load_data(train_path)
train_df = feature_engineering(train_df)

X_total, y_total = split_features_target(train_df, TARGET, ID_COLUMN)
skf = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X_total, y_total))

study = tune_xgboost(train_df, splits)
print("Best AUC:", study.best_value)
print("Best params:", study.best_trial.params)

save_best_params(study, PARAMS_DIR, PARAMS_FILENAME)
# %%
