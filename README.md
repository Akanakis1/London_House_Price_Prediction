# London House Price Prediction: Regression-Based Price Modeling

This project models London property sale prices using a structured regression workflow with
location-aware feature engineering. The emphasis is on **clean preprocessing, benchmarking
against simple baselines, and interpretable performance gains**, rather than algorithmic complexity.

---

## Project Overview

**Objective**
Predict residential property sale prices in London and evaluate whether a regression model
can materially outperform simple statistical baselines and a linear model.

**Workflow**
Data cleaning → feature engineering → geospatial clustering → baseline benchmarking →
regression modeling → holdout evaluation → prediction export.

**Data**
`data/train.csv` (about 266,000 sales with prices) and `data/test.csv` (about 16,500 sales without prices).
These are the files used in the linked Kaggle notebook.

**Kaggle notebook**
https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction

---

## Why This Project Matters

- Demonstrates **end-to-end regression analysis** on real, messy property data
- Benchmarks the model against **mean, median, quantile and constant baselines**, and against a
  **linear (Ridge) model** on the same features
- Incorporates **location intelligence** via geospatial clustering
- Reports results in **business-relevant metrics** (£ MAE, R²) on a **holdout set that was never used
  for fitting, early stopping or model selection**

---

## Results

The data is split 80% train / 10% validation / 10% holdout (`random_state=42`). The validation set is used for
XGBoost early stopping and for choosing the best model. The holdout set is evaluated once, afterwards.

| Model                              | R² (Validation) | MAE (Validation) | R² (Holdout) | MAE (Holdout) | RMSE (Holdout) |
|------------------------------------|----------------:|-----------------:|-------------:|--------------:|---------------:|
| Constant Baseline                  | -0.2597         | £610,037         | -0.2994      | £607,361      | £1,265,355     |
| Mean Baseline                      | -0.0377         | £397,274         | -0.0429      | £393,529      | £1,133,605     |
| Median Baseline                    | -0.0353         | £397,212         | -0.0401      | £393,427      | £1,132,114     |
| Quantile Baseline (75th pct.)      | -0.0011         | £473,506         | -0.0015      | £468,681      | £1,110,880     |
| Ridge Regression (linear baseline) | 0.2821          | £234,952         | 0.3224       | £227,395      | £913,765       |
| **XGBoost Regression**             | **0.6331**      | **£133,179**     | **0.6515**   | **£128,774**  | **£655,273**   |

**Key takeaways**
- On the holdout set, XGBoost reaches **R² ≈ 0.65 and MAE ≈ £129K**. That is about 67% lower MAE than the mean
  baseline and about 43% lower than a linear model on the same features, so the gain is not only from the features.
- RMSE (£655K) is much larger than MAE (£129K). This suggests that a small number of very expensive
  properties drive most of the squared error. A natural next step is a per-price-band error analysis.
- Cluster-level price statistics used as features are computed from the training split only, so no
  validation or holdout price leaks into the features.

Metrics are also saved to `outputs/model_metrics.csv` when the script runs.

---

## Methodology

### Data Preparation
- Combined train and test datasets using an `is_train` flag to ensure consistent transformations
- Handled missing values:
  - categorical → `"Unknown"`
  - numerical → `0`, with a `_was_missing` indicator column for each, so the model can tell "missing"
    from "really zero"
- Dropped non-informative fields (e.g. constant country field)

### Feature Engineering
- Address decomposition (street, city, postcode)
- Temporal features (sale date, quarters, seasonality encoding)
- Size and density features (total rooms, room density)
- Log transformation of skewed variables (target price, floor area)
- Frequency encoding for high-cardinality columns (street, postcode, outcode), one-hot encoding for
  low-cardinality columns

### Location Intelligence
- Standardized latitude and longitude
- Applied **KMeans clustering (k=4)** to capture location-driven price patterns
- Added cluster-level price statistics (mean and median), computed from the training split only

### Modeling & Evaluation
- Split: 80% train / 10% validation / 10% holdout
- Benchmarks: mean, median, quantile and constant predictors, plus a Ridge regression on the same features
- Main model: gradient-boosted regression (XGBoost, early stopping on the validation set)
- Metrics reported after the inverse log transform (`expm1`) to preserve £ interpretability

### Limitations
- A single random split was used. Repeated or time-based splits would give a better view of stability
  (sale dates run over many years, so a time-based split would also test how the model handles drift).
- Hyperparameters were set by hand, not by systematic search.

---

## Repository Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── final/
│       └── London_Price_Predictions.csv
├── outputs/
│   ├── elbow_kmeans.png
│   └── model_metrics.csv
├── notebooks/
│   └── Exploratory_Data_Analysis_(EDA).ipynb
├── london_house_price_prediction.py
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/Akanakis1/London_House_Price_Prediction.git
   cd London_House_Price_Prediction
```

2. Install dependencies (scikit-learn 1.4+ and xgboost 2.0+ are required):
```bash
   pip install -r requirements.txt
```

3. Run the pipeline (it takes a few minutes, mostly XGBoost training):
```bash
   python london_house_price_prediction.py
```

Reported numbers were produced with Python 3.13, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1,
xgboost 3.4.1. Other library versions can shift the last digits.
