# 💂🏼‍♂️🇬🇧 London House Price Prediction – Advanced ML Techniques

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Project-blue?logo=kaggle)](https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction)
[![Python](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange?logo=xgboost)]

---

## 📊 Project Overview

This project focuses on predicting **property sale prices in London** using advanced machine learning techniques. Leveraging property features, geolocation, and sale timing, the aim is to build highly accurate and interpretable predictive models.

---

## 🔍 Motivation

Property prices in London fluctuate due to many factors, including location, house features, and market conditions. Accurate price predictions assist buyers, sellers, investors, and policymakers in making informed decisions. This project explores advanced feature engineering, spatial clustering, and gradient boosting to capture these complexities.

---

## 📘 Dataset Overview

The dataset contains detailed information on property sales across London. Each record represents a unique transaction.

### ✨ Key Variables

<div align="center">

| Variable              | Description                                      |
|-----------------------|-------------------------------------------------|
| `fullAddress`         | Complete property address including street, city, postal code |
| `postcode`            | Postal code of the property                      |
| `country`             | Country (e.g., "England")                        |
| `outcode`             | Outward postcode representing district/region   |
| `latitude`            | Geographic latitude                              |
| `longitude`           | Geographic longitude                             |
| `bathrooms`           | Number of bathrooms                              |
| `bedrooms`            | Number of bedrooms                               |
| `floorAreaSqM`        | Floor area in square meters                       |
| `livingRooms`         | Number of living rooms                           |
| `tenure`              | Ownership type (e.g., "Freehold", "Leasehold")  |
| `propertyType`        | Type of property (e.g., "Flat", "Detached House") |
| `currentEnergyRating` | Energy rating (e.g., "A", "B", "C", "None")     |
| `sale_month`          | Month of sale (1–12)                             |
| `sale_year`           | Year of sale                                    |
| `price`               | Sale price (target variable)                     |

</div>

---

## 🎯 Project Objective

Predict the **sale price** (`price`) of London properties using machine learning, focusing on:

- **Feature Engineering:** Extracting spatial, temporal, and property-based predictors  
- **Model Selection:** Testing multiple models to optimize accuracy  
- **Evaluation Metrics:**  
  - **R² (R-squared)** – Variance explained  
  - **MAE (Mean Absolute Error)** – Average error  
  - **MSE / RMSE** – Penalizes larger errors  

**Best Model:**  

- **XGBoost Regressor** with optimized hyperparameters and GPU acceleration.

---

## 🏆 Achievements

- Developed an advanced feature engineering pipeline, including geospatial clustering.  
- Built a robust predictive model with interpretable outcomes.  
- Provided insights on factors influencing London house prices.

---

## 🔧 Tools & Technologies

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **Platform:** Kaggle

---

## 📁 Repository Contents

<div align="center">

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `house_price_pipeline.py`      | End-to-end Python script for training, evaluation, and submission |
| `train.csv`                   | Training data with features and target                   |
| `test.csv`                    | Test data for predictions                                |
| `London_Price_Predictions.csv`| Submission file with model predictions                   |
| `requirements.txt`            | Python package dependencies                              |

</div>

---

## 📂 Project Directory Structure

London_House_Price_Prediction/  
├── .venv/  
├── .vscode/  
├── data/  
│ ├── final/  
│ │ └── London_Price_Predictions.csv  
│ ├── test.csv  
│ └── train.csv  
├── notebooks/  
│ ├── Exploratory_Data_Analysis_(EDA).ipynb  
├── london_house_price_prediction.py  
├── README.md  
├── requirements.txt  

- **data**: Raw and processed datasets, plus predictions  
- **data/final**: Model output files for submission  
- **notebooks**: Exploratory data analysis work  
- **house_price_pipeline.py**: Main executable script  
- **requirements.txt**: Dependencies  
- **README.md**: This documentation

---

## 🚀 Project Workflow Diagram

A[Load Data] -> B[Preprocessing & Cleaning]  
B  ->  C [Feature Engineering]  
C  ->  D [Geospatial Clustering (KMeans)]  
D  ->  E [Train/Validation Split]  
E  ->  F [Model Training (Baselines + XGBoost)]  
F  ->  G [Model Evaluation]  
G  ->  H [Best Model Selection]  
H  ->  I [Make Predictions]  
I  ->  J [Save Submission File]  

---
