# London House Price Prediction With Machine Learning
### Data Taken from: https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques

## Kaggle-projects
https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction

# About the Dataset
## Overview:

This dataset contains information about property sales in London, with each entry representing a sale transaction. Each sale includes detailed information such as the full address, geographical location, number of rooms, property type, energy rating, and sale price. The dataset includes both numeric and categorical features that can be used to predict the sale price of properties based on various factors, such as location, property type, and features of the house.

## The data has been split into two groups:

Inside the zip file, you can find the datasets.

- training set (train.csv)
- test set (test.csv)

## Columns:

- fullAddress: The full address of the property, including street name, city, and postal code.
- postcode: The postal code of the property.
- country: The country in which the property is located (e.g., "England").
- outcode: The outward part of the postcode, which typically represents a district or region.
- latitude: The geographical latitude of the property.longitude: The geographical longitude of the property.
- bathrooms: The number of bathrooms on the property. [Missing values are represented by NaN.]
- bedrooms: The number of bedrooms in the property. [Missing values are represented by NaN.]
- floorAreaSqM: The floor area of the property in square meters. [Missing values are represented by NaN.]
- livingRooms: The number of living rooms in the property. [Missing values are represented by NaN.]
- tenure: The type of ownership of the property, such as "Freehold" or "Leasehold". [Missing values are represented by NaN.]
- propertyType: The type of property, such as "Flat/Maisonette" or "Detached House". [Missing values are represented by NaN.]
- currentEnergyRating: The property's current energy rating, such as "A," "B," "C," "D," "E," "F," "G," or "None" if not available. [Missing values are represented by NaN.]
- sale_month: The month in which the property was sold (1–12).
- sale_year: The year in which the property was sold.
- price: The sale price of the property in GBP.

## Competition Objective:

Predict London house prices using monthly data, including property features, energy ratings, location, and historical sale data over time.

## Source:

Jake Wright. London House Price Prediction: Advanced Techniques. https://kaggle.com/competitions/london-house-price-prediction-advanced-techniques, 2024. Kaggle.
