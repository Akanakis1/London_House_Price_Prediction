# 💂🏼‍♂️🇬🇧 London House Price Prediction – Advanced ML Techniques

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Project-blue?logo=kaggle)](https://www.kaggle.com/code/alexandroskanakis/london-house-price-prediction)
[![Python](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange?logo=xgboost)]

---

## 📊 Project Overview

This project focuses on predicting **property sale prices in London** using advanced machine learning regression. The solution involved complex data preparation, including **log-transformation** of the target variable and **geospatial clustering** (KMeans) for enhanced feature engineering. The final model, an **XGBoost Regressor**, significantly outperformed multiple statistical baselines, demonstrating the power of gradient boosting on complex, structured data.

---

## 🎯 Project Objective

* Build a robust regression model to predict London house prices with maximum accuracy.
* Implement **log-transformation** of the target price to correct for distribution skew.
* Conduct **geospatial feature engineering** using KMeans clustering to segment the London market.
* Compare and validate performance against four statistical baselines (Mean, Median, Quantile, Constant).
* Select the best model based on the lowest **Mean Absolute Error (MAE)**.

---

## 🏆 Achievements & Results

### Model Evaluation Results

The **XGBoost Regressor** provided vastly superior results compared to all baselines on the validation set, validating the choice of advanced feature engineering and modeling.

<div align="center">
  
| Model Name | R^2 Score (Valid) | **Mean Absolute Error (MAE) Valid** | Root Mean Squared Error (RMSE) Valid |
| :--- | :--- | :--- | :--- |
| Mean Baseline | -0.0430 | 393,537.58 | 1,133,679.42 |
| Median Baseline | -0.0401 | 393,426.61 | 1,132,113.58 |
| Quantile Baseline | -0.0015 | 468,680.56 | 1,110,880.26 |
| Constant Baseline | -0.2994 | 607,360.79 | 1,265,355.27 |
| **XGBoost Regression** | **0.6548** | **128,308.44** | **652,160.64** |

</div>

**Key Achievements:**
* Achieved a strong $\text{R}^2$ of **$0.65$** and an MAE of **$£128,308$** with the XGBoost Regressor.
* Implemented **geospatial feature engineering** using KMeans to successfully segment and model location-based price variances.
* Successfully applied **log-transformation** and inverse-transformation to handle a highly skewed financial target variable.

---

## 🔧 Tools & Technologies

* **Programming Language:** Python
* **Libraries:** **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost** (Regresser)
* **Key Techniques:** Log Transformation, **KMeans Clustering**, Gradient Boosting (Regression)

---

## 📁 Repository Contents

<div align="center">
  
| File | Description |
| :--- | :--- |
| `london_house_price_prediction.py` | Full pipeline: preprocessing, modeling, evaluation, prediction |
| `train.csv`, `test.csv` | Dataset files |
| `data/final/London_Price_Predictions.csv` | Submission file with model predictions |
| `notebooks/Exploratory_Data_Analysis_(EDA).ipynb` | Notebook for initial data analysis and feature exploration |

</div>

---

## 🚀 Project Workflow Diagram

A[Load Data] $\rightarrow$ B[Preprocessing & Cleaning] $\rightarrow$ C[Feature Engineering] $\rightarrow$ D[Geospatial Clustering (KMeans)] $\rightarrow$ E[Train/Validation Split] $\rightarrow$ F[Model Training (Baselines + XGBoost)] $\rightarrow$ G[Model Evaluation] $\rightarrow$ H[Best Model Selection] $\rightarrow$ I[Predict on Test Set] $\rightarrow$ J[Save Submission File]
---
