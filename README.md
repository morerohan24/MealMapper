# Food Delivery Time Prediction

ML-based solution for predicting food delivery times using regression models.

## Problem
Predict delivery time in minutes for food orders based on partner info, locations, and order details.

## Dataset Features
- Delivery Partner Age & Rating
- Restaurant Location (Lat/Long)
- Delivery Location (Lat/Long)
- Order Type, Vehicle Type
- Target: Delivery Time (minutes)

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Train models:
```bash
python run_complete_pipeline.py
```

Run web app:
```bash
streamlit run app.py
```

## Project Structure
```
├── Dataset.csv
├── src/
│   ├── preprocessing.py
│   ├── eda_analysis.py
│   ├── model_training.py
│   └── prediction.py
├── models/
├── visualizations/
├── app.py
└── requirements.txt
```

## Models Trained
- Linear Regression
- Ridge/Lasso
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Best model selected based on RMSE/MAE metrics.
"# MealMapper" 
