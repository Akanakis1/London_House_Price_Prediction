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

### Fill missing numerical with 0
numerical_list = ["bathrooms", "bedrooms", "floorAreaSqM", "livingRooms"]
for col in numerical_list:
    house_df[col] = house_df[col].fillna(0)

## 3.2 Feature Engineering
### Split address into components
house_df[["street", "city", "postcode"]] = house_df["fullAddress"].str.rsplit(", ", n=2, expand=True)
house_df = house_df.drop(columns="fullAddress", axis=1)

### Drop country if only one unique value
if house_df["country"].nunique() == 1:
    house_df = house_df.drop(columns="country", axis=1)

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
### Frequency Encoding
for col in ["street", "city", "postcode", "outcode", "tenure", "propertyType", "currentEnergyRating"]:
    freq = house_df[col].value_counts(normalize=True)
    house_df[col + "_freq"] = house_df[col].map(freq)

### Label Encode "outcode"
house_df["outcode_encoded"] = house_df["outcode"].astype("category").cat.codes

### One-Hot Encoding
cat_cols = ["tenure", "propertyType", "currentEnergyRating", "outcode", "city"]
house_df = pd.get_dummies(house_df, columns=cat_cols, drop_first=True)

### Drop unnecessary columns
house_df = house_df.drop(columns=['street','postcode'], axis=1, errors='ignore')

### Clean feature names
house_df.columns = house_df.columns.str.replace(' ', '_')

## 3.4 Split Back into Train / Test
train_df = house_df[house_df['is_train'] == 1].drop(columns='is_train').reset_index(drop=True)
test_df = house_df[house_df['is_train'] == 0].drop(columns=['is_train','price']).reset_index(drop=True)

print(f"Shape of Training Dataset: {train_df.shape}")
print(f"Shape of Test Dataset: {test_df.shape}")


# =====================================================
# 4. Clustering Geo-Features
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

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(range(1,11))
plt.grid(True)
plt.show()

## 4.4 Fit Final KMeans (k=4 as chosen)
kmeans_geo = KMeans(n_clusters=4, n_init='auto', random_state=42)
train_df['geo_cluster'] = kmeans_geo.fit_predict(X_geo_train_scaled)
test_df['geo_cluster']  = kmeans_geo.predict(X_geo_test_scaled)

## 4.5 Compute & Merge Cluster Stats
cluster_stats = train_df.groupby('geo_cluster').agg(
    mean_price_geo_cluster=('price','mean'),
    median_price_geo_cluster=('price','median'),
).reset_index()

train_df = train_df.merge(cluster_stats, on='geo_cluster', how='left')
test_df  = test_df.merge(cluster_stats, on='geo_cluster', how='left')


# =====================================================
# 5. Model Training and Evaluation
# =====================================================
## 5.1 Define features and target
features = train_df.drop(columns=['ID','price','sale_date']).columns.tolist()
X = train_df[features]
y = train_df['price']

## 5.2 Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

## 5.3 Evaluation Function
def evaluate_model(model, X, Y):
    y_pred = model.predict(X)
    y_pred = np.exp(y_pred)     # inverse log-transform
    y_val = np.exp(Y)
    return {
        "R^2 Score": r2_score(y_val, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
        "Mean Squared Error": mean_squared_error(y_val, y_pred),
        "Root Mean Squared Error": root_mean_squared_error(y_val, y_pred)
    }

## 5.4 Baseline Models
models = {
    'Mean Baseline': DummyRegressor(strategy= "mean"),
    'Median Baseline': DummyRegressor(strategy= "median"),
    'Quantile Baseline': DummyRegressor(strategy= "quantile", quantile=0.75),
    'Constant Baseline': DummyRegressor(strategy= "constant", constant=0)
}

results = {}

### Train + Evaluate Baselines
for name, model in models.items():
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

## 5.5 XGBoost Regression (Main Model)
models = {
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
        device="cuda"
    )
}

results = {}

### Train + Evaluate XGBoost
for name, model in models.items():
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
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

## 5.6 Model Selection (Lowest MAE)
best_model_name = min(results, key=lambda name: results[name]['Validation']['Mean Absolute Error'])
best_model = models[best_model_name]
print(f"Best model selected: {best_model_name}")


# =====================================================
# 6. Submission File Creation
# =====================================================
def create_submission(best_model, test_df, features, id_col='ID', filename='London_Price_Predictions.csv'):
    """ Create submission file from best model predictions. """
    # Copy test set
    submission_df = test_df.copy()
    
    # Predictions (inverse log transform applied)
    submission_df['price'] = best_model.predict(submission_df[features])
    submission_df['price'] = np.exp(submission_df['price'])
    
    # Keep only ID + Price
    London_Price_Predictions = submission_df[[id_col, 'price']]
    
    # Save CSV
    output_dir = r'data\final'
    os.makedirs(output_dir, exist_ok=True)
    London_Price_Predictions.to_csv(os.path.join(output_dir, filename), index=False)
    
    print(f"Submission file saved as '{filename}'")

# Generate Final Submission
create_submission(best_model, test_df, features, id_col='ID', filename='London_Price_Predictions.csv')
