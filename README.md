# Adult Census Income Classification

## a. Problem statement

Predict whether a person earns more than $50K per year (`>50K`) or not (`<=50K`) from
demographic and employment attributes. This is a binary classification problem. Five
classifiers are trained and tuned on the same data, compared on six metrics, and served
through a Streamlit web app that evaluates the saved models on an uploaded test CSV.

## b. Dataset description

- **Source:** Adult Census Income (UCI ML Repository, via Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income)
- **Instances:** 32,561 raw rows; 32,537 after dropping 24 exact duplicates
- **Features:** 14 raw (6 numeric, 8 categorical) - exceeds the minimum of 12
- **Target:** `income` (`<=50K` = 0, `>50K` = 1); class balance is about 75.9% / 24.1%
- **Numeric:** age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
- **Categorical:** workclass, education, marital_status, occupation, relationship, race, sex, native_country
- **Split:** stratified 80/20 train/test (26,029 training rows, 6,508 test rows)

### Data preparation (from `eda.py`, plots in `eda/`)

| Check | Finding | Action |
|---|---|---|
| Missing values | `?` in workclass (5.64%), occupation (5.66%), native_country (1.79%) | Converted to NaN; imputed (most frequent for categorical, median for numeric) |
| Duplicates | 24 exact duplicate rows | Dropped |
| Redundant feature | `education` vs `education_num`: Cramér's V = 1.00 (same information) | Dropped `education`, kept the ordinal `education_num` |
| Uninformative feature | `fnlwgt` (census sampling weight): correlation with income = -0.009 | Dropped |
| Skew | capital_gain skew 11.95 (91.7% zeros), capital_loss skew 4.59 (95.3% zeros) | `log1p` transform, then standardise |
| Feature independence | Strongest categorical dependencies: relationship-sex 0.65, marital_status-relationship 0.49, marital_status-sex 0.46, occupation-sex 0.42 (Cramér's V). Numeric features are only weakly correlated with each other. | Kept (each adds signal; relationship and marital_status are the top predictors, V with target 0.45); this violates the independence assumption of Naive Bayes (see observations) |
| Class imbalance | 24.1% positive class | Balanced class weights were compared with no weights (see below): kept only for the SVM |
| Encoding / scaling | Categorical features are nominal | One-hot encoding (`handle_unknown="ignore"`); StandardScaler on numeric features |

**Cleaned test data.** `test_data.csv` (6,508 rows) is the cleaned test split: duplicates removed,
all 842 missing values filled with training-set statistics (mode / median, so no information leaks
from the test set), the unused columns `fnlwgt` and `education` removed, and the target written as
`<=50K` / `>50K`. It has 12 feature columns plus `income`, with no missing values. Scaling, log
transform and one-hot encoding are done inside the saved model pipelines (fit on the training split
only), so the cleaned CSV can be fed directly to every saved model.

**Class weights.** With the same data and features, `class_weight="balanced"` raised recall by
about 23-26 points but lowered precision by 16-20 points and accuracy by 4-6 points:

| Model | class_weight | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | None | 0.8454 | 0.7155 | 0.5950 | 0.6497 | 0.5555 |
| Logistic Regression | balanced | 0.7995 | 0.5564 | 0.8278 | 0.6655 | 0.5514 |
| Decision Tree | None | 0.8528 | 0.7529 | 0.5791 | 0.6547 | 0.5710 |
| Decision Tree | balanced | 0.7939 | 0.5471 | 0.8406 | 0.6628 | 0.5489 |
| SVM | None | 0.8506 | 0.7459 | 0.5765 | 0.6504 | 0.5648 |
| SVM | balanced | 0.8055 | 0.5637 | 0.8527 | 0.6787 | 0.5715 |

It did not improve MCC for Logistic Regression or the Decision Tree (it lowered it), so those two are
trained without class weights. For the SVM it improved MCC, F1 and AUC, so the SVM keeps it.
KNN and Gaussian Naive Bayes have no class-weight option.

## c. GitHub repository link

https://github.com/yogeshgola03/Adult-Census-Income-Classification


## d. Models used

Each model was tuned with 5-fold stratified `GridSearchCV` on the training split only, scored by
Matthews Correlation Coefficient (MCC), then evaluated once on the held-out test split (6,508 rows).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8454 | 0.8939 | 0.7155 | 0.5950 | 0.6497 | 0.5555 |
| Decision Tree | 0.8528 | 0.9023 | 0.7529 | 0.5791 | 0.6547 | 0.5710 |
| KNN | 0.8439 | 0.8898 | 0.6933 | 0.6314 | 0.6609 | 0.5608 |
| Naive Bayes | 0.7488 | 0.8660 | 0.4877 | 0.8463 | 0.6188 | 0.4891 |
| Support Vector Machine | 0.8055 | 0.8976 | 0.5637 | 0.8527 | 0.6787 | 0.5715 |

### Hyperparameter tuning

| Model | Search space | Best parameters | CV MCC | Train acc. | Test acc. |
|---|---|---|---|---|---|
| Logistic Regression | C in {0.01, 0.1, 1, 10} (no class weights) | C=1 | 0.551 | 0.846 | 0.845 |
| Decision Tree | max_depth {4, 6, 8, 10, 12}; min_samples_leaf {5, 20, 50}; criterion {gini, entropy} (no class weights) | depth=12, min_samples_leaf=50, gini | 0.590 | 0.865 | 0.853 |
| KNN | k {5, 11, 21, 31}; weights {uniform, distance}; p {1, 2} | k=21, p=1 (Manhattan), uniform | 0.573 | 0.861 | 0.844 |
| Naive Bayes (Gaussian) | var_smoothing: 10 values from 1e-9 to 1 | var_smoothing=0.01 | 0.508 | 0.756 | 0.749 |
| SVM (RBF) | C {0.5, 1, 3}; gamma {scale, 0.01}; class_weight {None, balanced} | C=1, gamma=scale, balanced | 0.578* | 0.818 | 0.806 |

\*The SVM grid search used a stratified 8,000-row subsample of the training data for speed, then the best
setting was refit on all training rows, so its CV score is not directly comparable with the others.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid, fast baseline: accuracy 0.845, precision 0.716 and the second-best AUC tier (0.894). Train and test accuracy are almost identical (0.846 vs 0.845), so there is no overfitting. Recall for the >50K class is modest (0.595) because of the 24% positive rate; balanced class weights raised recall but did not improve MCC, so the unweighted model is used. Highly interpretable through its coefficients. |
| Decision Tree | Best accuracy (0.853), AUC (0.902) and precision (0.753), and the best CV MCC. Depth 12 with `min_samples_leaf=50` fits the data well while the train-test accuracy gap stays small (1.2 points). Recall is the lowest (0.579), so it is the most conservative about predicting >50K. Trains in seconds and is easy to explain. |
| KNN | Mid-table (MCC 0.561). Manhattan distance with k=21 smooths out noise, and the train-test gap is 1.7 points. It is the slowest model to tune and has the largest saved file (about 18 MB) because it stores the training data; prediction time also grows with data size. |
| Naive Bayes | Weakest model (MCC 0.489), though much better than a run on the raw features with default settings (accuracy 0.52, MCC 0.31), thanks to dropping redundant columns and `var_smoothing=0.01`. It still over-predicts >50K (recall 0.85, precision 0.49). Its independence assumption is violated by strongly associated features (relationship-sex V=0.65, marital_status-relationship V=0.49), and one-hot columns are binary, not Gaussian. |
| Support Vector Machine | Best F1 (0.679) and highest recall (0.853), with the (marginally) best MCC (0.5715). Balanced class weights make it favour catching high earners, at the cost of precision (0.56) and accuracy (0.806); for this model they improved MCC, F1 and AUC. It has the slowest training time among the single fits, and its search was run on a subsample. |
| **Overall winner for this dataset?** | **Support Vector Machine, narrowly.** It has the highest MCC (0.5715) and F1 (0.679) and the best recall, but its MCC edge over the Decision Tree (0.5710) is far smaller than split-to-split noise, so the two are effectively tied. The Decision Tree is better on accuracy, AUC and precision, is far faster to train and is interpretable. The choice depends on the goal: the SVM if missing a high earner is costly, the Decision Tree if accuracy, precision and explainability matter. All results come from a single train/test split. |

## Streamlit app

1. Upload a test CSV (or tick "Use bundled test_data.csv"). The file needs the 12 feature columns of
   `test_data.csv` plus `income`; extra columns (for example the original `fnlwgt` and `education`) are ignored.
2. The first 20 rows are shown in the preview, with a note giving how many rows the file has in total.
3. Press **Run all models on test data** to generate the results and Pick a model in the dropdown to see results.
4. Results appear in two tabs: the **selected model** (six metrics, confusion matrix, classification report, ROC curve)
   and **all models** (comparison table with the best value per metric highlighted, heatmap and combined ROC curves).

Models are loaded from `model/` and only used for prediction; nothing is retrained on the uploaded data.

## How to run

```bash
pip install -r requirements.txt
python eda.py adult.csv              # optional: EDA, writes plots to eda/
python train_models.py adult.csv     # tunes + trains, writes model/ and test_data.csv (about 5 min)
streamlit run app.py                 # launches the app
```

## Repository structure

```
app.py              Streamlit app
common.py           cleaning, feature lists and metric helpers
eda.py              exploratory analysis and independence checks
train_models.py     tuning, training, evaluation and cleaned test-set export
requirements.txt
test_data.csv       cleaned held-out test split (12 features + income)
eda/                EDA plots
model/              tuned pipelines (*.joblib), metrics.csv, best_params.json, tuning_summary.csv
```
