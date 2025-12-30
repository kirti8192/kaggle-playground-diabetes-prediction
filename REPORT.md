# Diabetes Prediction (Binary Classification) Report

## Problem and Metric
This project predicts whether a patient is diagnosed with diabetes based on tabular clinical and demographic features. It is a binary classification task evaluated with ROC-AUC, which is appropriate under class imbalance and avoids committing to a fixed decision threshold.

Challenge URL: https://www.kaggle.com/competitions/playground-series-s5e12/overview/abstract

## Data Summary (Modeling-Relevant Only)
The training set has 700,000 rows and 26 columns (including `id` and the target), and the test set has 300,000 rows and 25 columns. Features are a mix of numerical and categorical variables, with no missing values. These properties favor tree-based models that can handle heterogeneous feature types and large sample sizes efficiently.

## Feature Engineering (Motivation and Scope)
A small set of ratio and interaction features was added based on clinical intuition and derived from routinely available measurements:
- Pulse pressure: systolic minus diastolic blood pressure.
- Non-HDL cholesterol: total cholesterol minus HDL.
- LDL/HDL ratio.
- Triglycerides/HDL ratio.
- Total cholesterol/HDL ratio.
- Physical activity minutes per week divided by BMI.

These are well-known clinical metrics and easy to obtain; they were included because they encode non-linear relationships already implicit in the raw inputs. Broad feature expansion was avoided to keep the model stable, interpretable, and computationally reasonable given the dataset size.

## Model Selection Rationale
CatBoost and LightGBM were chosen because they natively handle categorical features and scale well to large tabular datasets. Earlier experiments with ordinal and one-hot encoding introduced artificial orderings or high-dimensional sparse features, which can degrade performance or increase training cost without clear gains. In particular, ordinal encoding for XGBoost imposed a ranking on categories (e.g., education level) that is not inherently ordered with respect to diabetes probability, so the encoding was not meaningful. Native categorical handling produced a simpler pipeline with more stable validation behavior.

## Cross-Validation Strategy
Stratified K-fold cross-validation was used to preserve class proportions across folds and obtain reliable generalization estimates. Out-of-fold (OOF) predictions were generated for each model; these are required to evaluate blends without bias. Using OOF predictions for ensembling prevents leakage that would otherwise inflate performance if the ensemble weights were tuned on in-fold predictions.

## Hyperparameter Tuning
Optuna was used with a constrained search space after feature engineering stabilized. Tuning was run in Kaggle notebooks for convenience and reproducibility. This kept the search focused on meaningful trade-offs (e.g., tree complexity, regularization) without overfitting to noisy hyperparameter combinations. Exhaustive searches were avoided and early stopping was used to manage runtime.

## Ensemble Method
The final ensemble is a linear blend of CatBoost and LightGBM probability outputs. Blend weights were selected using OOF ROC-AUC rather than any leaderboard feedback. The motivation is error diversity: each model makes different mistakes, and a calibrated blend tends to reduce variance and smooth model-specific failure modes. Because the weights are fit on OOF predictions, the blend is less likely to overfit and is expected to generalize better than selecting a single model based on a single split.

## Figures and Diagnostics
Figures will be specified later.

## Learnings
What worked:
- Native categorical handling simplified preprocessing and improved validation stability.
- Stratified CV with OOF predictions made model comparison and ensembling reliable.
- A small, motivated set of ratio features added signal without inflating the feature space.

What did not:
- Aggressive feature expansion via one-hot encoding was not efficient at this scale and added complexity without clear gains.

## Concepts and Tools Used (Scripts)
- Data handling and EDA: pandas/numpy workflows for sanity checks, class balance, and basic distribution inspection.
- Validation and metrics: Stratified K-fold CV, ROC-AUC evaluation, out-of-fold predictions.
- Feature processing: ordinal encoding, one-hot encoding, native categorical handling.
- Models: XGBoost, CatBoost, LightGBM classifiers.
- Optimization and regularization: early stopping, constrained hyperparameter search, Optuna-based tuning.
- Ensembling: linear blending of model probabilities and weight selection using OOF AUC.

## References
- Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features."
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
