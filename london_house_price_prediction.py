# =====================================================
# 1. Import Libraries
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import os

# =====================================================
# 2. Data Loading
# =====================================================
train_df = pd.read_csv(r"data\train.csv")
test_df = pd.read_csv(r"data\test.csv")

# Add flag before merging datasets
train_df['is_train'] = 1
test_df['is_train'] = 0

print(f"Shape of Training Dataset: {train_df.shape}")
print(f"Shape of Test Dataset: {test_df.shape}")

# =====================================================
# 3. Data Preprocessing
# =====================================================
house_df = pd.concat([train_df, test_df], axis=0)

## 3.1 Data Cleaning
### Fill missing categorical with "Unknown"
cat_fill_cols = ["tenure", "propertyType", "currentEnergyRating"]
for col in cat_fill_cols:
    house_df[col] = house_df[col].fillna("Unknown")

### Fill missing numerical with 0 — but first record WHICH rows were missing.
# FIX: filling with 0 alone makes "legitimately near-zero" and "was missing"
# indistinguishable to the model. A boolean flag lets XGBoost recover that
# signal if it turns out to be predictive.
numerical_list = ["bathrooms", "bedrooms", "floorAreaSqM", "livingRooms"]
for col in numerical_list:
    house_df[col + "_was_missing"] = house_df[col].isna().astype(int)
    house_df[col] = house_df[col].fillna(0)

## 3.2 Feature Engineering
### Split address into components
house_df[["street", "city", "postcode"]] = house_df["fullAddress"].str.rsplit(", ", n=2, expand=True)
house_df = house_df.drop(columns="fullAddress")

### Drop country if only one unique value
# Normalize whitespace/case first so near-duplicate values (e.g. trailing
# spaces) don't cause nunique() to report more than 1 and silently skip the
# drop, which is what happened here — "country" survived as an unencoded
# string column and broke XGBoost's DMatrix construction later.
house_df["country"] = house_df["country"].astype(str).str.strip()
if house_df["country"].nunique() == 1:
    house_df = house_df.drop(columns="country")

### Apply log-transform to skewed variables
house_df[['price', 'floorAreaSqM']] = house_df[['price', 'floorAreaSqM']].clip(lower=0)
house_df[['price', 'floorAreaSqM']] = np.log1p(house_df[['price', 'floorAreaSqM']])

### Time-based features
house_df['sale_date'] = pd.to_datetime(house_df['sale_year'].astype(str) + '-' + house_df['sale_month'].astype(str) + '-01')
house_df['days_since_first_sale'] = (house_df['sale_date'] - house_df['sale_date'].min()).dt.days
house_df['sale_quarter'] = house_df['sale_date'].dt.quarter
house_df['sale_month_sin'] = np.sin(2 * np.pi * house_df['sale_month'] / 12)
house_df['sale_month_cos'] = np.cos(2 * np.pi * house_df['sale_month'] / 12)

### Room features
house_df['total_rooms'] = house_df['bedrooms'] + house_df['bathrooms'] + house_df['livingRooms']
house_df['room_density'] = house_df['floorAreaSqM'] / (house_df['total_rooms'] + 1)

## 3.3 Encoding for Model Readiness
# FIX: previously every one of tenure/propertyType/currentEnergyRating/outcode/city
# was frequency-encoded AND one-hot-encoded (and outcode ALSO label-encoded) —
# three overlapping representations of the same signal for the same column.
# Split columns by cardinality instead and pick ONE encoding per column:
#   - high-cardinality (street, postcode, outcode) -> frequency encoding
#   - low-cardinality (tenure, propertyType, currentEnergyRating, city) -> one-hot

### Frequency Encoding — high-cardinality columns only
high_card_cols = ["street", "postcode", "outcode"]
for col in high_card_cols:
    freq = house_df[col].value_counts(normalize=True)
    house_df[col + "_freq"] = house_df[col].map(freq)

### One-Hot Encoding — low-cardinality columns only
low_card_cols = ["tenure", "propertyType", "currentEnergyRating", "city"]
house_df = pd.get_dummies(house_df, columns=low_card_cols, drop_first=True)

### Drop raw high-cardinality columns now that they're frequency-encoded
house_df = house_df.drop(columns=["street", "postcode", "outcode"], errors='ignore')

### Clean feature names
house_df.columns = house_df.columns.str.replace(' ', '_')

## 3.4 Split Back into Train / Test
train_df = house_df[house_df['is_train'] == 1].drop(columns='is_train').reset_index(drop=True)
test_df = house_df[house_df['is_train'] == 0].drop(columns=['is_train', 'price']).reset_index(drop=True)

print(f"Shape of Training Dataset: {train_df.shape}")
print(f"Shape of Test Dataset: {test_df.shape}")

# =====================================================
# 4. Geo Clustering (unsupervised — no target used, safe pre-split)
# =====================================================
## 4.1 Define geo features
geo_features = ['latitude', 'longitude']

## 4.2 Standardize coordinates
scaler = StandardScaler()
X_geo_train_scaled = pd.DataFrame(scaler.fit_transform(train_df[geo_features]), columns=geo_features, index=train_df.index)
X_geo_test_scaled = pd.DataFrame(scaler.transform(test_df[geo_features]), columns=geo_features, index=test_df.index)

## 4.3 Elbow Method for optimal k
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_geo_train_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

## 4.4 Fit Final KMeans (k=4 as chosen)
# NOTE: geo_cluster assignment itself only uses latitude/longitude — no target
# (price) is involved here, so fitting this on the full training set before the
# train/validation split does not leak target information. Only the PRICE
# STATISTICS computed per cluster (section 6 below) can leak, so those are
# deliberately computed after the split, using the training portion only.
kmeans_geo = KMeans(n_clusters=4, n_init='auto', random_state=42)
train_df['geo_cluster'] = kmeans_geo.fit_predict(X_geo_train_scaled)
test_df['geo_cluster'] = kmeans_geo.predict(X_geo_test_scaled)

# =====================================================
# 5. Train-Validation Split
# =====================================================
## 5.1 Define features and target
# NOTE: cluster price-statistic columns don't exist yet at this point —
# they're added in section 6, strictly after this split.
features = train_df.drop(columns=['ID', 'price', 'sale_date']).columns.tolist()
X = train_df[features]
y = train_df['price']

## 5.2 Train-Validation Split (done BEFORE any target-derived feature is built)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# =====================================================
# 6. Cluster Price Statistics — LEAKAGE-SAFE VERSION
# =====================================================
# Fit is done ONLY on the training split (X_train / y_train). The same
# per-cluster statistics are then applied to X_val and test_df via merge.
# Validation rows never contribute to the statistics used to describe them.
train_with_target = X_train.copy()
train_with_target['price'] = y_train.values
cluster_stats = train_with_target.groupby('geo_cluster').agg(
    mean_price_geo_cluster=('price', 'mean'),
    median_price_geo_cluster=('price', 'median'),
).reset_index()

# Global fallback in case a cluster in val/test wasn't seen in this particular
# training split (unlikely with k=4 and this dataset size, but safe to have).
global_mean_price = y_train.mean()
global_median_price = y_train.median()

def attach_cluster_stats(df):
    merged = df.merge(cluster_stats, on='geo_cluster', how='left')
    merged['mean_price_geo_cluster'] = merged['mean_price_geo_cluster'].fillna(global_mean_price)
    merged['median_price_geo_cluster'] = merged['median_price_geo_cluster'].fillna(global_median_price)
    return merged

X_train = attach_cluster_stats(X_train)
X_val = attach_cluster_stats(X_val)
test_df = attach_cluster_stats(test_df)

# Keep the features list in sync with the two new columns
features = X_train.columns.tolist()

# =====================================================
# 6b. Safety Check — catch leftover non-numeric columns BEFORE modeling
# =====================================================
# XGBoost requires numeric/bool/category dtypes. If any preprocessing step
# above didn't fully encode/drop a column, this will flag it loudly instead
# of failing deep inside XGBoost's DMatrix construction with a confusing
# traceback.
non_numeric_cols = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
if non_numeric_cols:
    print(f"WARNING: dropping unencoded non-numeric columns before modeling: {non_numeric_cols}")
    X_train = X_train.drop(columns=non_numeric_cols)
    X_val = X_val.drop(columns=non_numeric_cols)
    test_df = test_df.drop(columns=non_numeric_cols)
    features = [f for f in features if f not in non_numeric_cols]

# =====================================================
# 7. Model Training and Evaluation
# =====================================================
## 7.1 Evaluation Function
def evaluate_model(model, X, Y):
    y_pred = model.predict(X)
    # FIX: price was log-transformed with np.log1p, so the inverse must be
    # np.expm1 (not np.exp) to undo it exactly. On this dataset the gap was
    # tiny (~0.01% relative error, since prices start at £10,000) — but it
    # was still the wrong inverse, and would matter more on smaller values.
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(Y)
    return {
        "R^2 Score": r2_score(y_true, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Mean Squared Error": mean_squared_error(y_true, y_pred),
        "Root Mean Squared Error": root_mean_squared_error(y_true, y_pred)
    }

## 7.2 Candidate Models — baselines AND the main model live in ONE dict now.
# FIX: previously `results`/`models` were reset to {} right before the
# XGBoost block, so "Model Selection (Lowest MAE)" below could only ever
# return XGBoost — the baseline numbers were computed, printed, then
# discarded before selection happened. Keeping everything in one dict makes
# the selection step do what its name says.
models = {
    'Mean Baseline': DummyRegressor(strategy="mean"),
    'Median Baseline': DummyRegressor(strategy="median"),
    'Quantile Baseline': DummyRegressor(strategy="quantile", quantile=0.75),
    'Constant Baseline': DummyRegressor(strategy="constant", constant=0),
    "XGBoost Regression": xgb(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.05,
        min_child_weight=6,
        reg_alpha=0.5,
        reg_lambda=5,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        device="cpu",  # don't hard-require a GPU to reproduce results
        early_stopping_rounds=50,  # FIX: eval_set was being passed to .fit()
        # for logging only — nothing was actually stopping training early,
        # so all 1500 trees were built regardless of validation performance
        # at max_depth=10. This now picks the best iteration automatically.
    )
}

### Train + Evaluate
results = {}
for name, model in models.items():
    if name == "XGBoost Regression":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    else:
        model.fit(X_train, y_train)

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results[name] = {"Train": train_metrics, "Validation": val_metrics}

    print(f"Model: {name}")
    print("Training set evaluation:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 50)
    print("Validation set evaluation:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 50, "\n")

## 7.3 Model Selection (Lowest MAE) — now genuinely compares all candidates
best_model_name = min(results, key=lambda name: results[name]['Validation']['Mean Absolute Error'])
best_model = models[best_model_name]
print(f"Best model selected: {best_model_name}")

# =====================================================
# 8. Submission File Creation
# =====================================================
def create_submission(best_model, test_df, features, id_col='ID', filename='London_Price_Predictions.csv'):
    """ Create submission file from best model predictions. """
    # Copy test set
    submission_df = test_df.copy()
    # Predictions (inverse log transform applied)
    submission_df['price'] = best_model.predict(submission_df[features])
    submission_df['price'] = np.expm1(submission_df['price'])  # FIX: matches log1p
    # Keep only ID + Price
    London_Price_Predictions = submission_df[[id_col, 'price']]
    # Save CSV
    output_dir = r'data\final'
    os.makedirs(output_dir, exist_ok=True)
    London_Price_Predictions.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Submission file saved as '{filename}'")

# Generate Final Submission
create_submission(best_model, test_df, features, id_col='ID', filename='London_Price_Predictions.csv')
