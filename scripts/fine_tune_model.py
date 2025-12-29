# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from datetime import datetime
import matplotlib.pyplot as plt

# %%
# params
SEED = 42
TARGET = 'diagnosed_diabetes'
N_FOLDS = 5
TOP_N_IMPORTANCE = 30  # plot only top N feature names (otherwise unreadable)

# %%
# Load dataset
df_train_total = pd.read_csv('../data/raw/playground-series-s5e12/train.csv')
df_test = pd.read_csv('../data/raw/playground-series-s5e12/test.csv')

# %%
# feature engineering
def feature_engineering(df):
    eps = 1e-6

    df = df.copy()  # avoid mutating caller
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['non_hdl'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['ldl_hdl'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + eps)
    df['tg_hdl'] = df['triglycerides'] / (df['hdl_cholesterol'] + eps)
    df['total_hdl'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + eps)
    df['activity_bmi'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + eps)

    return df

# %%
# apply feature engineering
df_train_total = feature_engineering(df_train_total)
df_test = feature_engineering(df_test)

# %%
# make X and y for total training data
X_total = df_train_total.drop(columns=[TARGET, 'id'])
y_total = df_train_total[TARGET].astype(int)  # minor safety

# %%
# evaluation metric
def evaluate(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# %%
# CV split
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_eval_results = []
feature_names_per_fold = []
models = []                   
test_pred_folds = []          

for fold_idx, (train_index, val_index) in enumerate(skf.split(X_total, y_total), start=1):

    # get this fold's train and val dataframes
    df_train = df_train_total.iloc[train_index]
    df_val = df_train_total.iloc[val_index]

    # get X and y
    X_train = df_train.drop(columns=[TARGET, 'id']).copy() 
    y_train = df_train[TARGET].astype(int)

    X_val = df_val.drop(columns=[TARGET, 'id']).copy()
    y_val = df_val[TARGET].astype(int)

    X_test = df_test.drop(columns=['id']).copy()

    # ordinal features
    ordinal_cols = [
        'education_level',
        'income_level',
        'smoking_status',
    ]

    # ohe features
    ohe_cols = [
        'gender',
        'ethnicity',
        'employment_status',
    ]

    # ordinal encoding (explicit ordering)
    ordinal_encoder = OrdinalEncoder(
        categories=[
            ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
            ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
            ['Current', 'Former', 'Never'],
        ]
    )
    X_train[ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])
    X_val[ordinal_cols] = ordinal_encoder.transform(X_val[ordinal_cols])
    X_test[ordinal_cols] = ordinal_encoder.transform(X_test[ordinal_cols])

    # one hot encoding
    ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_ohe = ohe_encoder.fit_transform(X_train[ohe_cols])
    X_val_ohe = ohe_encoder.transform(X_val[ohe_cols])
    X_test_ohe = ohe_encoder.transform(X_test[ohe_cols])

    # build feature names that match the concatenated arrays
    base_feature_names = X_train.drop(columns=ohe_cols).columns.tolist()
    ohe_feature_names = ohe_encoder.get_feature_names_out(ohe_cols).tolist()
    feature_names = base_feature_names + ohe_feature_names
    feature_names_per_fold.append(feature_names)

    # concatenate one hot encoded features
    X_train = np.concatenate([X_train.drop(columns=ohe_cols).to_numpy(), X_train_ohe], axis=1)
    X_val = np.concatenate([X_val.drop(columns=ohe_cols).to_numpy(), X_val_ohe], axis=1)
    X_test_arr = np.concatenate([X_test.drop(columns=ohe_cols).to_numpy(), X_test_ohe], axis=1)

    # xgboost model -- fresh instance per fold
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED,

        max_depth=4,
        min_child_weight=3,
        gamma=0,

        n_estimators=500,
        learning_rate=0.1,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_lambda=1.0,
        reg_alpha=0.0,

        tree_method="hist",
        early_stopping_rounds=100,
    )

    # train model
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # evaluate on validation set
    y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
    val_auc = evaluate(y_val, y_val_pred)
    print(f'Fold {fold_idx} Validation AUC: {val_auc:.4f}')

    # store eval history for plotting
    fold_eval_results.append(xgb_model.evals_result())

    # store model + optional test preds
    models.append(xgb_model)
    test_pred_folds.append(xgb_model.predict_proba(X_test_arr)[:, 1])

# %%
# plot validation AUC curves for all folds
plt.figure(figsize=(10, 6))
for fold_idx, evals_result in enumerate(fold_eval_results, start=1):
    val_auc_curve = evals_result['validation_0']['auc']
    plt.plot(range(1, len(val_auc_curve) + 1), val_auc_curve, label=f'Fold {fold_idx}')
plt.xlabel('Boosting Round')
plt.ylabel('Validation AUC')
plt.title('Validation AUC Curves per Fold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# plot feature importance (from last fold as representative)
xgb_model_last = models[-1]
feature_names_last = feature_names_per_fold[-1]

feature_importances = xgb_model_last.feature_importances_
if len(feature_importances) != len(feature_names_last):
    raise ValueError(
        f"Mismatch: importances={len(feature_importances)} vs names={len(feature_names_last)}. "
        "This indicates a preprocessing/feature-name alignment issue."
    )

# top-N plot for readability (FIX)
top_idx = np.argsort(feature_importances)[::-1][:TOP_N_IMPORTANCE]
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_idx))[::-1], feature_importances[top_idx][::-1], align='center')
plt.yticks(range(len(top_idx))[::-1], [feature_names_last[i] for i in top_idx][::-1])
plt.xlabel('Feature Importance')
plt.title(f'XGBoost Feature Importance (Top {TOP_N_IMPORTANCE}, last fold)')
plt.tight_layout()
plt.show()

# %%
# predict on test set (CV-ensemble average; minimal upgrade, safe) (OPTIONAL BUT GOOD)
y_test_pred = np.mean(np.vstack(test_pred_folds), axis=0)

# %%
# save submission
submission = pd.DataFrame({
    'id': df_test['id'],
    TARGET: y_test_pred
})

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'../submissions/submission_{timestamp}_cv.csv'
submission.to_csv(filename, index=False)

print("Saved:", filename)
# %%
# re-train on full training data and ignore 10 least important features (upgrade attempt)
# identify least important features from last fold
feature_importances = xgb_model_last.feature_importances_
least_important_idx = np.argsort(feature_importances)[:10]
features_to_drop = [feature_names_last[i] for i in least_important_idx]
print("Dropping least important features:", features_to_drop)

# re-encode categorical features on full data
# ordinal encoding
ordinal_cols = [
    'education_level',
    'income_level',
    'smoking_status',
]
ordinal_encoder = OrdinalEncoder(
    categories=[
        ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
        ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
        ['Current', 'Former', 'Never'],
    ]
)
X_total[ordinal_cols] = ordinal_encoder.fit_transform(X_total[ordinal_cols])
X_test[ordinal_cols] = ordinal_encoder.transform(X_test[ordinal_cols])

# one hot encoding
ohe_cols = [
    'gender',
    'ethnicity',
    'employment_status',
]
ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_total_ohe = ohe_encoder.fit_transform(X_total[ohe_cols])
X_test_ohe = ohe_encoder.transform(X_test[ohe_cols])

# concatenate one hot encoded features
X_total_reduced = np.concatenate([X_total.drop(columns=ohe_cols).to_numpy(), X_total_ohe], axis=1)
X_test_reduced = np.concatenate([X_test.drop(columns=ohe_cols).to_numpy(), X_test_ohe], axis=1)

# prepare full training data
X_total_reduced = X_total.drop(columns=features_to_drop).copy()
X_test_reduced = df_test.drop(columns=['id'] + features_to_drop).copy()

# re-train model on full reduced data
xgb_model_final = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=SEED,

    max_depth=4,
    min_child_weight=3,
    gamma=0,

    n_estimators=500,
    learning_rate=0.1,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,
    reg_alpha=0.0,

    tree_method="hist",
)

xgb_model_final.fit(
    X_total_reduced,
    y_total,
    verbose=100
)

# predict on test set with final model
y_test_pred_final = xgb_model_final.predict_proba(X_test_reduced)[:, 1]

# %%
# save submission
submission = pd.DataFrame({
    'id': df_test['id'],
    TARGET: y_test_pred_final
})

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'../submissions/submission_{timestamp}_fulltrain_dropleastimpfeatures.csv'
submission.to_csv(filename, index=False)

print("Saved:", filename)