# London House Price Prediction With Machine Learning
### Data Taken from: https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques

## Kaggle-projects
https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction

## Overview
This dataset contains information about property sales in London, with each entry representing a sale transaction. Each property sale has detailed information such as the full address, geographical location, number of rooms, property type, energy rating, and the sale price. The dataset includes both numeric and categorical features that can be used to predict the sale price of properties based on various factors such as location, property type, and features of the house.

## The data has been split into two groups:

Inside the zip file, you can find the datasets.

- training set (train.csv)
- test set (test.csv)

## Columns
- fullAddress

Type: String

Description: The full address of the property, including street name, city, and postal code.

- postcode

Type: String

Description: The postal code of the property.

- country

Type: String
  
Description: The country in which the property is located (e.g., "England").

- outcode

Type: String

Description: The outward part of the postcode, which typically represents a district or region.

- latitude

Type: Float

Description: The geographical latitude of the property.

- longitude

Type: Float

Description: The geographical longitude of the property.

- bathrooms

Type: Float

Description: The number of bathrooms in the property. Missing values are represented by NaN.

- bedrooms

Type: Float

Description: The number of bedrooms in the property.

- floorAreaSqM

Type: Float

Description: The floor area of the property is in square meters.

- livingRooms

Type: Float

Description: The number of living rooms in the property. Missing values are represented by NaN.

- tenure

Type: String

Description: The type of ownership of the property, such as "Freehold" or "Leasehold".

- propertyType

Type: String

Description: The type of property, such as "Flat/Maisonette" or "Detached House".

- currentEnergyRating

Type: String

Description: The current energy rating of the property, such as "A", "B", "C", or "None" if not available.

- sale_month

Type: Integer

Description: The month in which the property was sold (1–12).

- sale_year

Type: Integer

Description: The year in which the property was sold.

- price

Type: Float

Description: The sale price of the property is GBP.
