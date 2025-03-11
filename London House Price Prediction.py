# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as  MAE


# Data Importing
train_df = pd.read_csv('C:/Users/alexa/OneDrive/Έγγραφα/Work_Python/Data_Analysis/Project 3 London House Price Prediction/train.csv')
test_df = pd.read_csv('C:/Users/alexa/OneDrive/Έγγραφα/Work_Python/Data_Analysis/Project 3 London House Price Prediction/test.csv')

## Concatenation of DataFrames
house_df = pd.concat([train_df, test_df], axis=0)

# Data Inspection
def inspect(df, name="Dataset"):
    print(f"Head {name}:\n", df.head(), "\n")
    print(f"Shape of {name}:\n", df.shape, "\n")
    print(f"Info of {name}:\n")
    df.info()
    print(f"\nSummary {name}:\n", df.describe(include='all'))
    print(f"Missing values in {name}:\n", df.isna().sum(), "\n")
inspect(train_df, "Train dataset")
inspect(test_df, "Test dataset")
inspect(house_df, "All House Dataset")

# Data Preprocess
## Log Transformation
house_df['price'] = np.log(house_df['price'])
house_df['floorAreaSqM'] = np.log(house_df['floorAreaSqM'])

## Splitting Full Address
house_df[['street', 'city', 'postcode']] = house_df['fullAddress'].str.rsplit(', ', expand = True, n = 2)

## Filling NaN Values for floorAreaSqM, bathrooms, bedrooms, and livingRooms
median_floor = house_df.groupby(['country', 'outcode'])['floorAreaSqM'].transform('median').round()
house_df['floorAreaSqM'] = house_df['floorAreaSqM'].fillna(median_floor)

median_bath = house_df.groupby(['floorAreaSqM'])['bathrooms'].transform('median').round()
median_house_bath = house_df.groupby(['country'])['bathrooms'].transform('median').round()
median_bath = median_bath.fillna(median_house_bath)
house_df['bathrooms'] = house_df['bathrooms'].fillna(median_bath)

median_bed = house_df.groupby(['floorAreaSqM'])['bedrooms'].transform('median').round()
median_house_bed = house_df.groupby(['country'])['bedrooms'].transform('median').round()
median_bed = median_bed.fillna(median_house_bed)
house_df['bedrooms'] = house_df['bedrooms'].fillna(median_bed)

median_livi = house_df.groupby(['floorAreaSqM'])['livingRooms'].transform('median').round()
median_house_livi = house_df.groupby(['country'])['livingRooms'].transform('median').round()
median_livi = median_livi.fillna(median_house_livi)
house_df['livingRooms'] = house_df['livingRooms'].fillna(median_livi)

## Feature Mapping for Regions
regions_map = {
    'Leytonstone':1, 'Walthamstow':1, 'Leyton':1, 'Stratford':1, 'Chingford':1,
    'Forest Gate':1, 'Woodford Green':1, # East London
    'London':2, # Central London
    'Southgate':3, 'Wembley':3,'Edmonton':3, 'Palmers Green':3, # North London
    'Blackheath':4, 'Woolwich':4, 'Kidbrooke':4, 'Charlton':4, 'Abbey Wood':4,
    'Greenwich':4, 'Eltham':4,  'Deptford':4, # South East London
    'Wimbledon':5, 'Raynes Park':5, 'Colliers Wood':5, # South West London
    'Acton':6, 'West Ealing':6, 'Ealing':6, 'Hanwell':6, 'Chiswick':6, 'Park Royal':6, # West London
    'Plumstead':7,'Bromley':7 # South London
    }
house_df['regions'] = house_df['city'].map(regions_map)

## Feature Mapping from more Expensive Outcodes to Less
outcode_map = {
    'W1B': 4, 'W1C': 4, 'W1D': 4, 'W1F': 4, 'W1G': 4, 'W1H': 4, 'W1J': 4, 'W1K': 4, 'W1S': 4, 'W1T': 4,
    'W1U': 4, 'W1W': 4, 'SW1A': 4, 'SW1E': 4, 'SW1H': 4, 'SW1P': 4, 'SW1V': 4, 'SW1W': 4, 'SW1X': 4, 'SW1Y': 4,
    'WC1A': 4, 'WC1B': 4, 'WC1E': 4, 'WC1H': 4, 'WC1N': 4, 'WC1R': 4, 'WC1V': 4, 'WC1X': 4,'WC2A': 4, 'WC2B': 4,
    'WC2E': 4, 'WC2H': 4, 'WC2N': 4, 'WC2R': 4, 'W10':4, # Most Expensive (Prime Central London)
    'EC1A': 3, 'EC1M': 3, 'EC1N': 3, 'EC1R': 3, 'EC1V': 3, 'EC1Y': 3, 'EC2A': 3, 'EC2M': 3, 'EC2N': 3, 'EC2R': 3,
    'EC2V': 3, 'EC2Y': 3, 'EC3A': 3, 'EC3M': 3, 'EC3N': 3, 'EC3R': 3, 'EC3V': 3, 'EC4A': 3, 'EC4M': 3, 'EC4R': 3,
    'EC4V': 3, 'EC4Y': 3, 'SW3': 3, 'SW5': 3, 'SW6': 3, 'SW7': 3, 'SW10': 3, 'SW11': 3, 'W2': 3, 'W8': 3,
    'W9': 3, 'W11': 3, 'W14': 3, # Expensive (Central London and Prime Areas)   
    'N1': 2, 'N2': 2, 'N3': 2, 'N4': 2, 'N5': 2, 'N6': 2, 'N7': 2, 'N8': 2, 'N10': 2, 'N11': 2,
    'N12': 2, 'N13': 2, 'N14': 2, 'N15': 2, 'N16': 2, 'N19': 2, 'N20': 2, 'N21': 2, 'N22': 2,'NW1': 2,
    'NW2': 2, 'NW3': 2, 'NW5': 2, 'NW6': 2, 'NW8': 2, 'NW10': 2, 'NW11': 2, 'SE1': 2, 'SE10': 2, 'SE11': 2,
    'SE15': 2, 'SE16': 2, 'SE21': 2, 'SE22': 2, 'SE24': 2, 'SW2': 2, 'SW4': 2, 'SW8': 2, 'SW9': 2, 'SW12': 2,
    'SW13': 2, 'SW14': 2, 'SW15': 2, 'SW16': 2, 'SW17': 2, 'SW18': 2, 'SW19': 2, 'SW20': 2, 'W3': 2, 'W4': 2,
    'W5': 2, 'W6': 2, 'W7': 2, 'W12': 2, 'W13': 2, # Moderately Expensive (Outer Central London and Suburban Prime Areas)
    'E1': 1, 'E2': 1, 'E3': 1, 'E4': 1, 'E5': 1, 'E6': 1, 'E7': 1, 'E8': 1, 'E9': 1, 'E10': 1,
    'E11': 1, 'E12': 1, 'E13': 1, 'E14': 1, 'E15': 1, 'E16': 1, 'E17': 1, 'E18': 1, 'E1W': 1,'IG8': 1,
    'N9': 1, 'N17': 1, 'N18': 1, 'NW4': 1, 'NW7': 1, 'NW9': 1,'SE2': 1, 'SE3': 1, 'SE4': 1, 'SE5': 1,
    'SE6': 1, 'SE7': 1, 'SE8': 1, 'SE9': 1, 'SE12': 1, 'SE13': 1, 'SE14': 1, 'SE17': 1, 'SE18': 1, 'SE19': 1,
    'SE20': 1, 'SE23': 1, 'SE25': 1, 'SE26': 1, 'SE27': 1, 'SE28': 1 # Less Expensive (Outer London and Suburban Areas)
    }
house_df['expen_outcode'] = house_df['outcode'].map(outcode_map)

## Handling Categorical Features
house_df['city'] = house_df['city'].astype('category').cat.codes

house_df['outcode'] = house_df['outcode'].astype('category').cat.codes

house_df['currentEnergyRating'] = house_df['currentEnergyRating'].fillna('Unknown').astype('category').cat.codes

house_df['tenure'] = house_df['tenure'].fillna('Unknown').astype('category').cat.codes

house_df['propertyType'] = house_df['propertyType'].fillna('Unknown').astype('category').cat.codes

season_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
house_df['seasons'] = house_df['sale_month'].map(season_map)

## Split house_df back into train_df and test_df
train_df = house_df.iloc[:len(train_df)].reset_index(drop=True)
test_df = house_df.iloc[len(train_df):len(train_df) + len(test_df)].reset_index(drop=True)

train_df = train_df[[
    'ID', 'price',
    'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
    'regions', 'city', 'street', 'postcode', 'outcode', 'expen_outcode', 'latitude', 'longitude',
    'currentEnergyRating', 'propertyType', 'tenure',
    'sale_month','sale_year', 'seasons'
    ]]
test_df = test_df[[
    'ID',
    'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
    'regions', 'city', 'street', 'postcode', 'outcode', 'expen_outcode', 'latitude', 'longitude',
    'currentEnergyRating', 'propertyType', 'tenure',
    'sale_month','sale_year', 'seasons'
    ]]


# # Visualazation with Pairplots
# sns.pairplot(train_df[[
#     'price',
#     'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
#     'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
#     'currentEnergyRating', 'propertyType', 'tenure',
#     'sale_month','sale_year', 'seasons'
# ]])
# sns.pairplot(test_df[[
#     'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
#     'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
#     'currentEnergyRating', 'propertyType', 'tenure',
#     'sale_month','sale_year', 'seasons'
# ]])
# plt.show()

# # Histogram Visualazation
# train_df[[
#     'price',
#     'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
#     'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
#     'currentEnergyRating', 'propertyType', 'tenure',
#     'sale_month','sale_year', 'seasons',
# ]].hist(bins=42)
# test_df[[
#     'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
#     'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
#     'currentEnergyRating', 'propertyType', 'tenure',
#     'sale_month','sale_year', 'seasons'
# ]].hist(bins=42)
# plt.show()

# ## Correlation Visualization
# correlation = train_df[[
#     'price',
#     'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
#     'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
#     'currentEnergyRating', 'propertyType', 'tenure',
#     'sale_month','sale_year', 'seasons'
# ]].corr()
# # Create a figure heatmap for Correlation
# fig, ax = plt.subplots(1, figsize=(16, 6))
# sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
# ax.set_title('Correlation between the variables', fontsize=16, fontweight='bold', fontname='Times New Roman')
# # plt.tight_layout()
# # plt.show()

# ## Covariance Visualization
# covariance = train_df[[
#     'price',
#     'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
#     'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
#     'currentEnergyRating', 'propertyType', 'tenure',
#     'sale_month','sale_year', 'seasons'
# ]].cov() # Compute the covariance matrix
# # Create a figure heatmap for Covariance
# fig, ax = plt.subplots(1, figsize=(16, 6))
# sns.heatmap(covariance, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
# ax.set_title('Covariance between the variables', fontsize=16, fontweight='bold', fontname='Times New Roman')
# plt.tight_layout()
# plt.show()

# Define features and target
features = [
    'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM',
    'regions', 'city', 'outcode', 'expen_outcode', 'latitude', 'longitude',
    'currentEnergyRating', 'propertyType', 'tenure',
    'sale_month','sale_year', 'seasons'
    ]
X = train_df[features]
y = train_df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Mean Squared Error': MSE(y_test, y_pred),
        'Root Mean Squared Error': np.sqrt(MSE(y_test, y_pred)),
        'Mean Absolute Error': MAE(y_test, y_pred)
    }

models = {
    'XGBoost Regression': xg.XGBRegressor(),
    'AdaBoost Regression': AdaBoostRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor()
}

## Train each model only once!
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # Train once!
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = metrics
    print(f"Model : {name}")
    for metric, value in metrics.items():
        print(f"{metric} : {value:.4f}")
    print('_____________________________________________')

## Determine the best model based on RMSE
best_model_name = min(results, key=lambda name: results[name]['Mean Absolute Error'])
best_model = models[best_model_name]
print(f"Best model selected: {best_model_name}")

## Now, connect (or “chain”) your create_submission function with the best model.
def create_submission(best_model, test_df, features, id_col='ID', filename='submission.csv'):
    submission_df = test_df.copy()
    submission_df['price'] = best_model.predict(submission_df[features])
    submission_df['price'] = np.exp(submission_df['price'])
    submission = submission_df[[id_col, 'price']]
    submission.to_csv(filename, index=False)
    print(f"Submission file saved as '{filename}'")

## Call create_submission with the best model and your test dataset
create_submission(best_model, test_df, features, id_col='ID', filename='submission.csv')