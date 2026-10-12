# Bacterial Growth Parameter Regression

### Predicting cyanobacteria growth curves from experimental conditions with four regression pipelines

Coursework project for Newcastle University's MSc Data Science & AI programme (module **CSC8635**, mini-project 2). It predicts four growth-curve parameters of bacteria (`a`, `mu`, `tau`, `a0`) from the conditions of each experiment, and compares linear regression, ridge regression, support vector regression and a random forest, with grid-search tuning inside cross-validation.

📓 **Notebook (code and written report):** [`notebooks/Bacterial_Growth_Regression.ipynb`](notebooks/Bacterial_Growth_Regression.ipynb)

---

## At a Glance

| | |
|---|---|
| **Task** | Regression: predict four growth parameters, one model per target |
| **Data** | 19,992 experiments, 5 input features, 4 target variables |
| **Pipelines** | Linear regression, ridge, SVR, random forest (per target) |
| **Evaluation** | 5-fold cross-validation with grid search inside each fold; RMSE, MAE, R² |
| **Best pipeline** | Random forest for all four targets |
| **Best scores** | R² of 0.979 (`a`), 0.977 (`mu`), 0.993 (`tau`), 0.708 (`a0`) |
| **Framework** | Python, scikit-learn |

## Table of Contents

1. [Background](#1-background)
2. [Problem Statement and Brief](#2-problem-statement-and-brief)
3. [Dataset](#3-dataset)
4. [Solution: Approach](#4-solution-approach)
5. [Implementation](#5-implementation)
6. [Results](#6-results)
7. [Limitations and Future Work](#7-limitations-and-future-work)
8. [Repository Structure](#8-repository-structure)
9. [Setup and Running](#9-setup-and-running)

---

## 1. Background

Bacterial growth depends strongly on the experimental environment: how much CO₂ and light are available, how many cells the culture starts with, and how well the strain produces sucrose. Predicting growth from these conditions helps with experimental design and biological analysis. Each experiment's growth curve is summarised by four parameters, and the question is how well they can be predicted, and with what kind of model, from the conditions alone.

## 2. Problem Statement and Brief

The CSC8635 brief asked for the following on the regression dataset:

1. Exploratory data analysis (EDA).
2. Data preprocessing.
3. A train/test/validation strategy.
4. **Four** machine-learning pipelines for predicting `a`, `mu`, `tau` and `a0`, **including hyper-parameter tuning**. This means four pipelines evaluated for each target, not one pipeline per target.
5. Results presented with appropriate metrics.
6. A short report on the approaches used, which worked best, and why.

The brief also warned that when predicting one target, the other three must **not** be used as features, because that would make the task trivially easy and meaningless. The written report sits inside the notebook, as required.

## 3. Dataset

`fitting-results.csv`, provided through the university's course platform: **19,992 rows × 9 columns**, no missing values.

| Variable | Role | Description | Range / mean |
|---|---|---|---|
| `n_cyanos` | Feature | Number of cyanobacteria at the start | 2 to 1,000 |
| `co2` | Feature | CO₂ available | 0.001 to 100 |
| `light` | Feature | Light available | 0 to 1 |
| `SucRatio` | Feature | How well the bacteria produce sucrose | 0 to 1 |
| `Nsample` | Feature | Experiment number (an identifier) | 1 to 19,995 |
| `a` | **Target** | Maximum number of bacteria seen | mean 5.49, std 13.44, max 162.2 |
| `mu` | **Target** | Growth rate | mean 5.63, std 9.86, max 123.4 |
| `tau` | **Target** | Time delay before growth starts | mean 0.373, std 0.421 |
| `a0` | **Target** | Initial level of bacteria | mean 0.9916, std 0.0171 |

`a0` is almost constant (median 1.0), while `a` and `mu` are strongly right-skewed. The input features differ widely in scale.

> **Not included** in this repository. The notebook expects `fitting-results.csv` (see [Setup and Running](#9-setup-and-running)).

## 4. Solution: Approach

### 4.1 Preprocessing and leakage control

- Features are `n_cyanos`, `co2`, `light`, `SucRatio` and `Nsample`. The four targets are removed from the feature matrix, and each target is predicted separately, so no target information leaks into the inputs.
- Preprocessing sits **inside every pipeline**: median imputation (a safeguard, since there are no missing values) followed by standardisation. Parameters are therefore learned from training folds only.

### 4.2 The four pipelines

| # | Pipeline | Tuned hyper-parameters (grid) |
|---|---|---|
| 1 | Linear regression | None (no relevant hyper-parameters) |
| 2 | Ridge regression | `alpha` ∈ {0.1, 1.0, 10.0} |
| 3 | Support vector regression (SVR) | `C` ∈ {0.1, 1, 10}, `gamma` ∈ {`scale`, `auto`} |
| 4 | Random forest regressor | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20} |

The same four pipelines are run independently for each of the four targets, and they are kept in a single registry so they can be looped over uniformly.

### 4.3 Tuning and evaluation

- **Nested evaluation:** `GridSearchCV` (3-fold, scored on RMSE) is run **inside** each fold of an outer 5-fold `KFold` cross-validation (shuffled, `random_state=42`). Tuning therefore never sees the outer test fold.
- **Metrics:** RMSE (penalises large errors), MAE (typical error, less sensitive to outliers) and R² (share of variance explained), averaged over the outer folds.

## 5. Implementation

- **Language and libraries:** Python, scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `cross_validate`), pandas, NumPy, Matplotlib
- **Reproducibility:** fixed seed (`42`) for the cross-validation split and the random forest
- **Environment:** developed on Google Colab

## 6. Results

Mean over the five outer folds (lower RMSE and MAE are better; higher R² is better). Best result per target in bold.

### Target `a` (maximum number of bacteria)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear regression | 11.416 | 5.921 | 0.278 |
| Ridge | 11.416 | 5.919 | 0.278 |
| SVR | 8.694 | 1.819 | 0.582 |
| **Random forest** | **1.951** | **0.386** | **0.979** |

### Target `mu` (growth rate)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear regression | 8.297 | 4.279 | 0.292 |
| Ridge | 8.297 | 4.277 | 0.292 |
| SVR | 6.076 | 1.350 | 0.620 |
| **Random forest** | **1.499** | **0.367** | **0.977** |

### Target `tau` (growth delay)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear regression | 0.2597 | 0.1725 | 0.620 |
| Ridge | 0.2597 | 0.1725 | 0.620 |
| SVR | 0.0858 | 0.0556 | 0.958 |
| **Random forest** | **0.0347** | **0.0189** | **0.993** |

### Target `a0` (initial level)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear regression | 0.0152 | 0.0104 | 0.213 |
| Ridge | 0.0152 | 0.0104 | 0.213 |
| SVR | 0.0510 | 0.0488 | −7.847 |
| **Random forest** | **0.0093** | **0.0039** | **0.708** |

The notebook also plots RMSE per model for each target.

### What the results mean

- **The random forest wins on every target.** It has the lowest RMSE and MAE and the highest R² for `a`, `mu`, `tau` and `a0`, which indicates that the relationship between experimental conditions and growth is strongly non-linear and involves interactions.
- **Linear models are a weak baseline.** Linear and ridge regression explain only about 21 to 29% of the variance for `a`, `mu` and `a0`, and ridge is indistinguishable from plain linear regression. Regularisation does not help when the problem is non-linearity rather than overfitting.
- **SVR sits in between.** It beats the linear models on `a`, `mu` and `tau` but does not match the forest.
- **`a0` is hardest.** Its values barely vary (standard deviation 0.017), so there is little signal to explain. SVR performs worse than predicting the mean (R² of −7.85). A plausible cause is that SVR's default ε-tube (0.1) is far larger than the whole spread of this target, but this was not tested in the notebook.
- **Targets behave differently**, which supports modelling each one separately rather than with a single shared model.

## 7. Limitations and Future Work

**Limitations**

- The hyper-parameter grids are small, to limit computation. A wider search, particularly for SVR, could change the ranking.
- The best hyper-parameters chosen inside each fold are not recorded in the notebook.
- `Nsample` (an experiment identifier) is used as a feature, as the notebook does, although it does not describe a physical quantity. No comparison without it was run.
- All targets share one preprocessing strategy. No target-specific transforms were tried, even though `a` and `mu` are strongly skewed.
- Only cross-validated scores are reported; there is no separate held-out test set.

**Future work**

- Wider or randomised hyper-parameter search, and gradient-boosted trees as an extra model.
- Log-transforming skewed targets (`a`, `mu`) and checking whether `Nsample` should be dropped.
- Feature-importance analysis of the random forest to say which conditions drive growth.
- For `a0`, test a wider SVR ε-tube or a different loss.

## 8. Repository Structure

```
bacterial-growth-regression/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── Bacterial_Growth_Regression.ipynb
```

The report is written inside the notebook, as the coursework required.

## 9. Setup and Running

The notebook was written for **Google Colab**: it mounts Google Drive and reads `fitting-results.csv` from a personal Drive folder. To run it elsewhere:

1. **Get the data.** Obtain `fitting-results.csv` (19,992 rows; columns `n_cyanos`, `co2`, `light`, `SucRatio`, `Nsample`, `a`, `mu`, `tau`, `a0`).
2. **Remove the Colab cell.** Delete the `google.colab` / `drive.mount(...)` cell. Set the path in the `pd.read_csv(...)` cell to your local file.
3. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the notebook top to bottom:** EDA → preprocessing → cross-validation and metrics → the four pipelines → grid-search evaluation for each target → plots and discussion.

The evaluation cells run grid search inside 5-fold cross-validation for four targets, so expect them to take a while on 20k rows.

---

*Author: Emaduddin Asdaq Syed Mohammed, School of Computing, Newcastle University.*
