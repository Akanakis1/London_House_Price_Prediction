# London House Price Prediction: Advanced Machine Learning Techniques

---

## 📁 Dataset Overview

This project aims to predict house prices in London using a variety of machine learning techniques. The dataset contains valuable features, such as property characteristics, geographic location, energy ratings, and sale history. The goal is to use these features to develop a robust model that can accurately predict house prices.

### 📊 Data Source  
[Kaggle London House Price Dataset](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques)

### 🔗 Kaggle Project  
Check out my solution and code here:  
**[London House Price Prediction – Alexandros Kanakis](https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction)**

The dataset includes information about various property sales in London, with each entry representing a single sale transaction.
Below are the key features provided:

## 📘 Data Dictionary

### ✨ Key Variables

- **`fullAddress`**: Full property address, including street name, city, and postal code.
- **`postcode`**: The postal code of the property.
- **`country`**: The country where the property is located (e.g., "England").
- **`outcode`**: The outward part of the postcode, typically representing a district or region.
- **`latitude`**: The geographical latitude of the property.
- **`longitude`**: The geographical longitude of the property.
- **`bathrooms`**: Number of bathrooms in the property.
- **`bedrooms`**: Number of bedrooms in the property.
- **`floorAreaSqM`**: Property floor area in square meters.
- **`livingRooms`**: Number of living rooms in the property.
- **`tenure`**: The type of ownership (e.g., "Freehold" or "Leasehold").
- **`propertyType`**: Type of property (e.g., "Flat", "Detached House").
- **`currentEnergyRating`**: Energy rating of the property (e.g., "A", "B", "C", or "None").
- **`sale_month`**: The month in which the property was sold (1–12).
- **`sale_year`**: The year in which the property was sold.
- **`price`**: The final sale price of the property (target variable).

## 🎯 Project Objective

The primary goal of this project is to predict the sale price (`price`) of properties in London based on the available features. We'll use several machine learning algorithms to build, train, and evaluate predictive models:

- **LassoCV (Lasso Regression with Cross-Validation)**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

The models will be evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

## 🛠️ Installation (Optional)

```bash
git clone https://github.com/Akanakis1/London_House_Price_Prediction.git
cd Titanic_Machine_Learning_from_Disaster
pip install -r requirements.txt  # if provided
python Titanic.ipynb     # or run via Jupyter Notebook