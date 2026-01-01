# Diabetes Prediction (Binary Classification) Report

## Problem and Metric
This project predicts whether a patient is diagnosed with diabetes based on tabular clinical and demographic features. It is a binary classification task evaluated with ROC-AUC, which is appropriate under class imbalance and avoids committing to a fixed decision threshold. The competition is hosted on Kaggle: https://www.kaggle.com/competitions/playground-series-s5e12/overview/abstract

## Data Summary (Modeling-Relevant Only)
The training set contains 700,000 rows and 26 columns (including `id` and the target), while the test set has 300,000 rows and 25 columns. Features consist of numerical and categorical variables without missing values. These properties favor tree-based models that efficiently handle heterogeneous feature types and large sample sizes.

## Feature Engineering (Motivation and Scope)
A small set of ratio and interaction features was added based on routinely available measurements:
- Pulse pressure: \( \text{systolic blood pressure} - \text{diastolic blood pressure} \)
- Non-HDL cholesterol: \( \text{total cholesterol} - \text{HDL cholesterol} \)
- LDL/HDL ratio: \( \frac{\text{LDL cholesterol}}{\text{HDL cholesterol}} \)
- Triglycerides/HDL ratio: \( \frac{\text{triglycerides}}{\text{HDL cholesterol}} \)
- Total cholesterol/HDL ratio: \( \frac{\text{total cholesterol}}{\text{HDL cholesterol}} \)
- Physical activity minutes per week divided by BMI: \( \frac{\text{physical activity minutes/week}}{\text{BMI}} \)

While tree-based models can learn non-linear relationships implicitly, explicitly including these carefully chosen features improves model stability and data efficiency.

## Model Selection Rationale
XGBoost served as the baseline model but required categorical features to be encoded as either ordinal or one-hot, which imposes assumptions that may not hold, such as treating education level as ordinal when it is not strictly ordered for diabetes prediction. CatBoost and LightGBM were selected because they natively handle categorical features and scale efficiently to large tabular datasets. Native categorical handling simplifies the pipeline, avoids artificial orderings or sparse high-dimensional encodings, and yields more stable validation performance.

## Cross-Validation Strategy
Stratified 5-fold cross-validation was used to preserve class proportions across folds and provide reliable generalization estimates.

## Hyperparameter Tuning
Optuna was employed with Tree-structured Parzen Estimator (TPE) sampling to efficiently explore a constrained hyperparameter search space after feature engineering stabilized. Early stopping was applied to prevent overfitting and manage runtime.

## Ensemble Method
Out-of-fold (OOF) predictions were generated for each of the three models: XGBoost, CatBoost, and LightGBM. These OOF predictions enabled unbiased evaluation and blending. The final ensemble is a linear blend of the three models' probability outputs. Blend weights were selected via a direct three-weight (2 degrees of freedom) search optimizing OOF ROC-AUC. This approach leverages error diversity among models and reduces overfitting risk compared to tuning on in-fold predictions.

## Figures and Diagnostics

The following figures are included after this section:

**Figure 1. Validation ROC-AUC (mean ± 1 std across folds).**  
Comparison of XGBoost, LightGBM, and CatBoost showing the mean validation ROC-AUC across boosting iterations, with shaded regions indicating one standard deviation across cross-validation folds.

**Figure 2. Validation Logloss (mean ± 1 std across folds).**  
Comparison of XGBoost, LightGBM, and CatBoost showing the mean validation logloss across boosting iterations, with shaded regions indicating one standard deviation across cross-validation folds.

## Learnings
What worked:
- Native categorical handling simplified preprocessing and improved validation stability.
- A small, motivated set of ratio features added meaningful signal without inflating the feature space.
- Stratified cross-validation with OOF predictions enabled reliable model comparison and ensembling.
- Ensembling multiple models improved predictive performance over any single model.

What did not:
- Aggressive feature expansion via one-hot encoding was inefficient at this scale and added complexity without clear gains.

## Concepts and Tools Used (Scripts)
- Data handling and EDA: pandas and numpy workflows for sanity checks, class balance, and distribution inspection.
- Validation and metrics: Stratified K-fold CV, ROC-AUC evaluation, out-of-fold predictions.
- Feature processing: ordinal encoding, one-hot encoding, native categorical handling.
- Models: XGBoost, CatBoost, LightGBM classifiers.
- Optimization and regularization: early stopping, constrained hyperparameter search, Optuna-based tuning using TPE sampling.
- Ensembling: linear blending of model probabilities with weight selection using OOF ROC-AUC.

## References
- Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features."
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."

## Footnote: Leaderboard Context

At the end of the competition, the final submission achieved **Rank 1279** with a **ROC-AUC of 0.69496**, improving by **47 positions** relative to the public leaderboard.

For reference, the top-ranked submission (**Rank 1**, ROC-AUC **0.70504**) reported using an extensive hill-climbing ensemble involving **40+ models**. This comparison highlights the diminishing returns of increasingly complex ensembles: the gap between a small, well-validated blend of a few strong models and a large-scale ensemble is relatively modest, while the latter requires substantially more engineering effort and computational resources.

This project intentionally prioritizes **methodological clarity, reproducibility, and validation correctness** over leaderboard maximization, using out-of-fold predictions and simple linear blending rather than large-scale ensemble construction.