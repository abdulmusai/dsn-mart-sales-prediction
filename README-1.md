# DSN Bootcamp Qualification Hackathon 2026 — ML Track

Predicting total sales for a given product at a given DSN Mart store, using historical product and store attributes. This is a regression task submitted as part of the qualifying hackathon for the DSN AI Bootcamp.

## Problem

DSN Mart operates stores ranging from small corner shops to large hypermarkets across Nigeria. The goal is to predict `total_sales` for each product-store combination in `test.csv`, based on product attributes (weight, price, category, fat content, shelf visibility) and store attributes (age, size, location tier, format).

**Evaluation metric:** Root Mean Squared Error (RMSE) — lower is better.

## Approach

### 1. Exploratory Data Analysis
- Checked distribution of `total_sales` (right-skewed → log-transformed for modeling)
- Identified missing values in `product_weight_kg` and `store_size`
- Found `shelf_visibility` values of exactly 0, which are physically implausible for a listed product and were treated as missing
- Found inconsistent casing in `product_category` (e.g. `Frozen Foods` vs `frozen foods`) — normalized to a single canonical form

### 2. Data Cleaning
| Issue | Fix |
|---|---|
| Inconsistent `product_category` casing | Normalized via `.str.title()` + manual mapping |
| Missing `product_weight_kg` | Imputed using per-product-code mean, falling back to per-category mean |
| `shelf_visibility == 0` | Treated as missing, imputed with per-category mean |
| Missing `store_size` (entirely absent for 3 stores) | Kept as an explicit `"Missing"` category rather than guessed, since it wasn't recoverable from other rows of the same store |

### 3. Feature Engineering
- `price_per_kg` — product price normalized by weight
- `price_rank_in_category` — percentile rank of a product's price within its category, capturing relative positioning (budget vs premium) that raw price alone doesn't
- `store_age_bucket` — binned store age
- `category_group` — perishable vs non-perishable grouping
- `product_count_in_store`, `store_count_for_product` — popularity/assortment proxies
- Out-of-fold smoothed target encoding for `product_code`, `store_code`, and `product_category` (computed leak-free using K-fold, so training data never sees its own target when encoding itself)
- Dropped an earlier `category_avg_sales` feature that used the full training mean directly (mild target leakage); the OOF `product_category` target encoding already captures this signal safely

### 4. Modeling
- Trained and compared **LightGBM**, **CatBoost**, and **XGBoost** regressors on `log1p(total_sales)`, evaluated with proper 5-fold cross-validation (all three models trained and predicted within each fold, with out-of-fold predictions and test predictions accumulated across folds)
- CatBoost handled the largely categorical feature set natively and outperformed both LightGBM and XGBoost individually
- Searched blend weights across all three models; the best-performing blend was 100% CatBoost — LightGBM and XGBoost's errors overlapped too closely with CatBoost's to add anything on top

### 5. Results

| Model | CV RMSE (original scale) |
|---|---|
| XGBoost | ~1128 |
| LightGBM | ~1121 |
| **CatBoost (final)** | **~1110** |
| Blend of all three (best weights found: 100% CatBoost) | ~1110 |
| Naive mean baseline (reference) | ~1698 |

## Repository Structure

```
├── train.csv               # training data (product-store records with total_sales)
├── test.csv                # test data (predictions required)

├── pipeline.py            # improved pipeline: target encoding + CatBoost/LightGBM blend
├── submission.csv           # final predictions for Kaggle leaderboard
└── README.md
```

## How to Run

```bash
pip install lightgbm catboost
```

This produces `submission.csv` with columns `id, total_sales`, ready for upload to the competition leaderboard.

## Possible Next Steps
- Hyperparameter tuning of CatBoost specifically (depth, learning rate) via Optuna, since it's the clear individual leader
- Additional target-encoded interaction features (e.g. category × store_format)
- Model stacking (e.g. a linear meta-model on out-of-fold predictions) instead of simple weighted blending

## Author
Abdulrahman Musa Inuwa — Kano, Nigeria
