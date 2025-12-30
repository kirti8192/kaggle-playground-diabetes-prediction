# %%
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

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
PARAMS_FILENAME = "catboost_optuna_params.json"

CAT_COLS = [
    "education_level",
    "income_level",
    "smoking_status",
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
def build_model(seed, params):
    return CatBoostClassifier(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=params["random_strength"],
        bagging_temperature=params["bagging_temperature"],
        border_count=params["border_count"],
        random_seed=seed,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
        allow_writing_files=False,
    )


# %%
def tune_catboost(df_train_total, splits):
    X_total, y_total = split_features_target(df_train_total, TARGET, ID_COLUMN)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 254),
        }

        fold_scores = []
        for fold_idx, (train_index, val_index) in enumerate(splits, start=1):
            df_train = df_train_total.iloc[train_index]
            df_val = df_train_total.iloc[val_index]

            X_train = df_train.drop(columns=[TARGET, ID_COLUMN]).copy()
            y_train = df_train[TARGET].astype(int)
            X_val = df_val.drop(columns=[TARGET, ID_COLUMN]).copy()
            y_val = df_val[TARGET].astype(int)

            cat_feature_indices = [
                X_train.columns.get_loc(col) for col in CAT_COLS if col in X_train.columns
            ]

            model = build_model(SEED, params)
            model.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False,
            )

            y_val_pred = model.predict_proba(X_val)[:, 1]
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
        "CATBOOST_NUM_ITERATIONS": int(best_params["iterations"]),
        "CATBOOST_LEARNING_RATE": best_params["learning_rate"],
        "CATBOOST_DEPTH": int(best_params["depth"]),
        "CATBOOST_EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
        "CATBOOST_EVAL_METRIC": "AUC",
        "CATBOOST_LOSS_FUNCTION": "Logloss",
        "CATBOOST_L2_LEAF_REG": best_params["l2_leaf_reg"],
        "CATBOOST_RANDOM_STRENGTH": best_params["random_strength"],
        "CATBOOST_BAGGING_TEMPERATURE": best_params["bagging_temperature"],
        "CATBOOST_BORDER_COUNT": int(best_params["border_count"]),
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

study = tune_catboost(train_df, splits)
print("Best AUC:", study.best_value)
print("Best params:", study.best_trial.params)

save_best_params(study, PARAMS_DIR, PARAMS_FILENAME)
# %%
