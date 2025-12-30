# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from datetime import datetime

# %%
# params
SEED = 42
VAL_SIZE = 0.2
TARGET = 'diagnosed_diabetes'

# %%
# Load dataset
df_train_total = pd.read_csv('../data/raw/playground-series-s5e12/train.csv')
df_test = pd.read_csv('../data/raw/playground-series-s5e12/test.csv')

# %%
# train val split
df_train, df_val = train_test_split(
    df_train_total,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=df_train_total[TARGET]     # stratified split because target is imbalanced
)

# %%
# get X and y
X_train = df_train.drop(columns=[TARGET, 'id'])
y_train = df_train[TARGET]

X_val = df_val.drop(columns=[TARGET, 'id'])
y_val = df_val[TARGET]

X_test = df_test.drop(columns=['id'])

# %%
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

# ordinal encoding
ordinal_encoder = OrdinalEncoder(
        categories=[
        ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],    # natural order in education_level
        ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],  # natural order in income_level
        ['Current', 'Former', 'Never'],                             # natural order in smoking_status
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

# concatenate one hot encoded features
X_train = np.concatenate([X_train.drop(columns=ohe_cols).to_numpy(), X_train_ohe], axis=1)
X_val = np.concatenate([X_val.drop(columns=ohe_cols).to_numpy(), X_val_ohe], axis=1)
X_test = np.concatenate([X_test.drop(columns=ohe_cols).to_numpy(), X_test_ohe], axis=1)

# %%
# evaluation metric
def evaluate(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# %%
# xgboost model

xgb_model = XGBClassifier(
    objective="binary:logistic",    
    eval_metric="auc",          # retain auc as eval metric to maintain consistency with kaggle competition metric
    random_state=SEED,

    max_depth=4,
    min_child_weight=3,
    gamma=0,

    n_estimators=1000,
    learning_rate=0.1,

    subsample=0.8,          # row subsampling to reduce overfitting, drop % of data
    colsample_bytree=0.8,   # feature subsampling to reduce overfitting, drop % of features

    reg_lambda=1.0,         # L2 regularization on leaf weights
    reg_alpha=0.0,          # L1 regularization on leaf weights, can be used for feature selection

    tree_method="hist",     # faster tree construction algorithm for larger datasets, uses histogram approximation
    early_stopping_rounds=100,
)

# %%
# train model
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# %%
# evaluate on validation set
y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
val_auc = evaluate(y_val, y_val_pred)
print(f'Validation AUC: {val_auc:.4f}')

# %%
# plot auc over epochs
import matplotlib.pyplot as plt
results = xgb_model.evals_result()
plt.plot(results['validation_0']['auc'])
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('XGBoost AUC over Epochs')
plt.show()

# %%
# predict on test set
y_test_pred = xgb_model.predict_proba(X_test)[:, 1]

# %%
# save submission
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': y_test_pred
})

# timestamp based filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'../submissions/submission_{timestamp}.csv'
submission.to_csv(filename, index=False)

# %%
