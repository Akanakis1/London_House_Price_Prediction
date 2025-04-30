# London House Price Prediction: Advanced Machine Learning Techniques

## Overview

This project aims to predict house prices in London using a variety of machine learning techniques. The dataset contains valuable features, such as property characteristics, geographic location, energy ratings, and sale history. The goal is to use these features to develop a robust model that can accurately predict house prices.

## Dataset

The dataset includes information about various property sales in London, with each entry representing a single sale transaction. Below are the key features provided:

### Key Features

- **`fullAddress`**: Full property address, including street name, city, and postal code.
- **`postcode`**: The postal code of the property.
- **`country`**: The country in which the property is located (e.g., "England").
- **`outcode`**: The outward part of the postcode, typically representing a district or region.
- **`latitude`**: The geographical latitude of the property.
- **`longitude`**: The geographical longitude of the property.
- **`bathrooms`**: Number of bathrooms in the property.
- **`bedrooms`**: Number of bedrooms in the property.
- **`floorAreaSqM`**: Floor area of the property in square meters.
- **`livingRooms`**: Number of living rooms in the property.
- **`tenure`**: The type of ownership (e.g., "Freehold" or "Leasehold").
- **`propertyType`**: Type of property (e.g., "Flat", "Detached House").
- **`currentEnergyRating`**: Energy rating of the property (e.g., "A", "B", "C", or "None").
- **`sale_month`**: The month in which the property was sold (1–12).
- **`sale_year`**: The year in which the property was sold.
- **`price`**: The final sale price of the property (target variable).

## Objective

The primary goal of this project is to predict the sale price (`price`) of properties in London based on the available features. We'll use several machine learning algorithms to build, train, and evaluate predictive models:

- **LassoCV (Lasso Regression with Cross-Validation)**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

The models will be evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

## Installation

To set up this project, clone the repository and install the required dependencies by running:

```bash
pip install -r requirements.txt