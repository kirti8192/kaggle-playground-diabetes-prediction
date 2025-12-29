# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# %%
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"

SUBMISSION_DIR = "/kaggle/working"

SUBMISSION_PREFIX = "submission"
SUBMISSION_TAG = "catboost_cv"

# TRAIN_PATH = "../data/raw/playground-series-s5e12/train.csv"
# TEST_PATH = "../data/raw/playground-series-s5e12/test.csv"

# SUBMISSION_DIR = "../submissions"
# SUBMISSION_PREFIX = "submission"
# SUBMISSION_TAG = "catboost_cv"

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

CATBOOST_NUM_ITERATIONS = 20
CATBOOST_LEARNING_RATE = 0.1
CATBOOST_DEPTH = 6
CATBOOST_EARLY_STOPPING_ROUNDS = 100
CATBOOST_EVAL_METRIC = "AUC"
CATBOOST_LOSS_FUNCTION = "Logloss"
CATBOOST_L2_LEAF_REG = 3.0

SEED = 42
N_FOLDS = 5
TARGET = "diagnosed_diabetes"
ID_COLUMN = "id"

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
def build_model(seed):

    return CatBoostClassifier(
        task_type="GPU",
        devices="0",
        iterations=CATBOOST_NUM_ITERATIONS,
        learning_rate=CATBOOST_LEARNING_RATE,
        depth=CATBOOST_DEPTH,
        random_seed=seed,
        early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
        eval_metric=CATBOOST_EVAL_METRIC,
        verbose=EVAL_VERBOSE,
        loss_function=CATBOOST_LOSS_FUNCTION,
        l2_leaf_reg=CATBOOST_L2_LEAF_REG,
        allow_writing_files=False,
    )

# %%
def train_cv(df_train_total, df_test, seed, n_folds, target, id_column):

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

        model = build_model(seed)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_val, y_val),
            use_best_model=True,
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

    cv_results = train_cv(
        df_train_total,
        df_test,
        SEED,
        N_FOLDS,
        TARGET,
        ID_COLUMN,
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
