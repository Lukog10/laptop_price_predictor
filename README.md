# Laptop Price Predictor

> ML regression project to predict laptop prices (INR) using hardware and brand specifications.
> Best model: Random Forest Regressor — R² = 0.8857

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Results](#model-results)
- [Feature Engineering](#feature-engineering)
- [Streamlit App](#streamlit-app)
- [Author](#author)

## Overview
This project refactors the original exploratory notebook into a modular, production-ready Python package for laptop price prediction.
It includes reusable preprocessing utilities, model training scripts, inference helpers, unit tests, and a Streamlit web app for interactive predictions.

## Dataset
- Source: Kaggle — Laptop Price Dataset
- Rows: ~1300 | Features: 11
- Target: Price (INR), log-transformed during training

## Tech Stack
Python · pandas · scikit-learn · seaborn · matplotlib · Streamlit

## Project Structure
```text
laptop-price-predictor/
│
├── data/
│   └── laptop_data.csv
│
├── notebooks/
│   └── laptop_price_prediction_Multi_Model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── random_forest_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
│
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train all models
python -m src.train

# Launch Streamlit app
streamlit run app/streamlit_app.py
```

## Model Results
| Model             | R²     | MAE (INR) | RMSE (INR) |
|-------------------|--------|-----------|------------|
| Linear Regression | 0.8048 | 12,939.25 | 21,034.89  |
| Ridge Regression  | 0.8080 | 12,706.78 | 19,318.39  |
| Lasso Regression  | 0.8061 | 12,816.63 | 20,464.31  |
| Decision Tree     | 0.8190 | 11,395.95 | 18,949.56  |
| Random Forest     | 0.8857 | 9,140.72  | 14,032.06  |

## Feature Engineering
- Touchscreen and IPS features extracted from screen resolution text
- PPI computed from screen resolution and screen size
- Memory split into HDD / SSD / Hybrid / Flash Storage capacities, with low-impact storage features removed
- CPU categorized into Intel Core i3 / i5 / i7 / Other Intel Processor / AMD Processor
- GPU grouped by brand: Nvidia / Intel / AMD
- RAM and weight converted from raw strings to numeric values

## Streamlit App
Interactive web app for predicting laptop price from user-selected specs.
The app computes PPI automatically from screen size and resolution, then uses the saved Random Forest model to estimate price in INR.

## Author
Gokul R
GitHub: https://github.com/Lukog10
LinkedIn: https://linkedin.com/in/GokulRlukoG
