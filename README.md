# 🏠 Vietnamese House Price Prediction


This project aims to predict house prices in Vietnam using machine learning models, trained on a real estate dataset scraped from batdongsan.com and shared via Kaggle. The task is formulated as a regression problem.

📊 Dataset Overview
Contains over 30,000 samples with features such as Area, Number of Floors, Bedrooms, Bathrooms, Frontage, Road Width, and Legal Status.

Some features have high missing rates (e.g., Balcony Direction: 82%). Those with >50% missing were removed.

Key preprocessing steps:

Log transformation for skewed numerical features (Area, Frontage, Access Road).

Added binary indicators for missing values (e.g., Frontage_missing).

Created new feature: bathrooms_per_bedroom.

Applied standard scaling for numerical data and one-hot encoding for categorical ones.

🧠 Models Tested
Three models were implemented and compared:

## Linear Regression

Baseline model using only clean features.

**Performance on validation set:**

RMSE: ~1.847 billion VND

R²: ~0.28 (underfitting)

## XGBoost

Performed significantly better than linear regression.

Feature importance revealed new features (e.g., bathrooms_per_bedroom) were useful.

Chosen as the final model due to strong performance and faster training time compared to MLP.

**Performance on validation set:**
RMSE: ~1.740 billion VND

R²: ~0.367 

**Multi-layer Perceptron (MLP)**

Slightly worse than XGBoost in performance.

Higher computational cost.

Performance on validation set:
RMSE: ~1.780 billion VND

R²: ~0.337


## 🚀 Deployment
🔗 Web App (Streamlit)
Use the web interface to get real-time house price predictions:
👉 Streamlit App:https://xgboost-house-price-predictor-mpupw2sfci8fankzjjnckp.streamlit.app/

**🔌 API Endpoint**
Predict prices programmatically using a RESTful API:
Endpoint: https://xgboost-house-price-predictor.onrender.com/predict
Method: POST
Input JSON:

json
Copy
Edit
{
  "Area": 100.0,
  "Floors": 3,
  "Bedrooms": 4,
  "Bathrooms": 3,
  "Frontage": 5.0,
  "Access road": 6.0,
  "Legal status": "Have certificate"
}
**Response:**

{
  "prediction": 5.50
}
Value is in billion VND.

## 🔮 Future Work
Improve model performance by collecting higher-quality, feature-rich data (e.g., location, nearby amenities).



