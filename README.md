# London House Price Prediction — Regression-Based Price Modeling

This project models London property sale prices using a structured regression workflow with
location-aware feature engineering. The emphasis is on **clean preprocessing, benchmarking
against simple baselines, and interpretable performance gains**, rather than algorithmic complexity.

---

## Project Overview

**Objective**  
Predict residential property sale prices in London and evaluate whether a regression model
can materially outperform simple statistical baselines.

**Workflow**  
Data cleaning → feature engineering → geospatial clustering → baseline benchmarking →
regression modeling → prediction export.

**Kaggle notebook**  
https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction

---

## Why This Project Matters

- Demonstrates **end-to-end regression analysis** on real, messy property data
- Explicitly benchmarks ML performance against **mean, median, and quantile baselines**
- Incorporates **location intelligence** via geospatial clustering
- Reports results in **business-relevant metrics** (£ MAE, R²)

---

## Results (Validation Set)

Best model: **Gradient-Boosted Regression (XGBoost)**

| Model               | R² (Valid) | MAE (Valid) | RMSE (Valid) |
|---------------------|-----------:|------------:|-------------:|
| Mean Baseline       | -0.0430    | £393,538    | £1,133,679   |
| Median Baseline     | -0.0401    | £393,427    | £1,132,114   |
| Quantile Baseline   | -0.0015    | £468,681    | £1,110,880   |
| Constant Baseline   | -0.2994    | £607,361    | £1,265,355   |
| **XGBoost Regression** | **0.6548** | **£128,308** | **£652,161** |

**Key takeaway**  
The regression model achieves **R² ≈ 0.65** and **MAE ≈ £128K**, substantially outperforming
all statistical baselines.

---

## Methodology

### Data Preparation
- Combined train and test datasets using an `is_train` flag to ensure consistent transformations
- Handled missing values:
  - categorical → `"Unknown"`
  - numerical → `0`
- Dropped non-informative fields (e.g. constant country field)

### Feature Engineering
- Address decomposition (street, city, postcode)
- Temporal features (sale date, quarters, seasonality encoding)
- Size and density features (total rooms, room density)
- Log transformation of skewed variables (target price, floor area)

### Location Intelligence
- Standardized latitude and longitude
- Applied **KMeans clustering (k=4)** to capture location-driven price patterns
- Added cluster-level price statistics (mean and median)

### Modeling & Evaluation
- Train/validation split: 90/10
- Benchmarks: mean, median, quantile, constant predictors
- Main model: gradient-boosted regression
- Metrics reported after inverse log transform to preserve £ interpretability

---

## Repository Structure

├── data/  
│ ├── train.csv  
│ ├── test.csv  
│ └── final/  
│ └── London_Price_Predictions.csv  
├── notebooks/  
│ └── Exploratory_Data_Analysis_(EDA).ipynb  
├── london_house_price_prediction.py  
├── requirements.txt  
└── README.md  

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
