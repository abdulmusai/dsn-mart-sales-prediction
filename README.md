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
- `store_age_bucket` — binned store age
- `category_group` — perishable vs non-perishable grouping
- `product_count_in_store`, `store_count_for_product` — popularity/assortment proxies
- Out-of-fold smoothed target encoding for `product_code`, `store_code`, and `product_category` (computed leak-free using K-fold, so training data never sees its own target when encoding itself)

### 4. Modeling
- Trained and compared **LightGBM** and **CatBoost** regressors on `log1p(total_sales)`, evaluated with 5-fold cross-validation
- CatBoost handled the largely categorical feature set natively and outperformed LightGBM
- Searched blend weights between the two models; the best-performing blend was pure CatBoost

### 5. Results

| Model | CV RMSE (original scale) |
|---|---|
| LightGBM | ~1122 |
| **CatBoost (final)** | **~1108** |
| Naive mean baseline (reference) | ~1698 |

## Repository Structure

```
├── train.csv               # training data (product-store records with total_sales)
├── test.csv                # test data (predictions required)
├── pipeline.py              # baseline LightGBM pipeline
├── pipeline_v2.py            # improved pipeline: target encoding + CatBoost/LightGBM blend
├── submission.csv           # final predictions for Kaggle leaderboard
└── README.md
```

## How to Run

```bash
pip install lightgbm catboost
python pipeline_v2.py
```

This produces `submission.csv` with columns `id, total_sales`, ready for upload to the competition leaderboard.

## Possible Next Steps
- Hyperparameter tuning via Optuna
- Additional target-encoded interaction features (e.g. category × store_format)
- Model stacking instead of simple weighted blending

## Author
Abdulrahman — Kano, Nigeria
