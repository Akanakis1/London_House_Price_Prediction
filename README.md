# 💂🏼‍♂️🇬🇧 London House Price Prediction — Advanced ML Regression (XGBoost + Geo Clustering)

Predict **London property sale prices** using a full end-to-end ML workflow:  
**data cleaning → feature engineering → geospatial clustering → baseline benchmarking → XGBoost regression → submission file**.

🔗 Kaggle: https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction

---

## 🔥 Why this project stands out
- Built a **robust regression pipeline** with advanced feature engineering and validation benchmarking.
- Handled a highly skewed target using **log-transformation** + inverse transform for evaluation/predictions.
- Added **geospatial intelligence** with KMeans clustering (latitude/longitude) and cluster-level price statistics.
- Benchmarked against multiple **statistical baselines** to prove the ML lift (Mean/Median/Quantile/Constant).

---

## 🏆 Results (Validation Set)
Best model: **XGBoost Regressor**

| Model | R² (Valid) | MAE (Valid) | RMSE (Valid) |
|---|---:|---:|---:|
| Mean Baseline | -0.0430 | 393,537.58 | 1,133,679.42 |
| Median Baseline | -0.0401 | 393,426.61 | 1,132,113.58 |
| Quantile Baseline | -0.0015 | 468,680.56 | 1,110,880.26 |
| Constant Baseline | -0.2994 | 607,360.79 | 1,265,355.27 |
| **XGBoost Regression** | **0.6548** | **128,308.44** | **652,160.64** |

✅ Key takeaway: **R² ≈ 0.65** and **MAE ≈ £128K**, significantly outperforming statistical baselines.

---

## 🧠 Method (What I did)

### 1) Data preprocessing & cleaning
- Merged train + test using an `is_train` flag for consistent transformations.
- Filled missing values:
  - categorical → `"Unknown"` (e.g., tenure, propertyType, currentEnergyRating)
  - numerical → `0` (e.g., bedrooms, bathrooms, floorAreaSqM, livingRooms)

### 2) Feature engineering
- Split `fullAddress` into **street / city / postcode**.
- Dropped `country` if only one unique value.
- Applied **log1p transform** to `price` (target) and `floorAreaSqM` (skew handling).
- Time features:
  - `sale_date`, `days_since_first_sale`, `sale_quarter`
  - seasonal encoding: `sale_month_sin`, `sale_month_cos`
- Room density features:
  - `total_rooms`, `room_density`

### 3) Encoding strategy
- Frequency encoding: street, city, postcode, outcode, tenure, propertyType, energy rating.
- Label encoding: outcode
- One-hot encoding: tenure, propertyType, energy rating, outcode, city
- Dropped raw `street` and `postcode` after encoding (to control dimensionality)

### 4) Geospatial clustering (location-aware pricing)
- Standardized latitude/longitude using `StandardScaler`.
- Used elbow method (k=1..10) and selected **k=4**.
- Added cluster-level stats:
  - mean & median price per geo cluster

### 5) Model training & selection
- Train/validation split: 90/10, `random_state=42`.
- Baselines: DummyRegressor (mean/median/quantile/constant).
- Main model: **XGBoost Regressor** (GPU enabled via `device="cuda"`).
- Evaluation performed after inverse log transform to report metrics in original £ scale.

---

## 📁 Repository contents
- `london_house_price_prediction.py` — full pipeline: preprocessing → clustering → training → evaluation → submission
- `notebooks/Exploratory_Data_Analysis_(EDA).ipynb` — EDA and feature exploration
- `data/train.csv`, `data/test.csv` — dataset files
- `data/final/London_Price_Predictions.csv` — final predictions output

---

## 🚀 How to run

### 1) Install requirements
```bash
pip install -r requirements.txt
