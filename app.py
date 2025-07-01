import streamlit as st
import requests
#import os

# --- Configuration ---
# All this code is related to the user interface.
API_URL = "https://xgboost-house-price-predictor.onrender.com"
#API_HOST = os.getenv("API_HOST", "127.0.0.1")
#API_PORT = 8088  
#API_URL = f"http://{API_HOST}:{API_PORT}/predict"

st.title("🏠 House Price Prediction")
st.markdown("Enter the details of the property to get a price prediction.")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (m²)", min_value=10.0, value=100.0, step=1.0)
    floors = st.number_input("Number of Floors", min_value=1, value=1, step=1)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=2, step=1)
with col2:
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=1, step=1)
    frontage = st.number_input("Frontage (m)", min_value=1.0, value=5.0, step=0.5)
    access_road = st.number_input("Access Road Width (m)", min_value=1.0, value=5.0, step=0.5)
legal_status = st.selectbox("Legal Status", ["Sale contract", "Have certificate"])

# --- Prediction Logic ---
if st.button("Predict House Price", type="primary"):
    input_data = {
        "Area": area, "Floors": floors, "Bedrooms": bedrooms,
        "Bathrooms": bathrooms, "Frontage": frontage,
        "Access road": access_road, "Legal status": legal_status
    }
    with st.spinner('Sending data to the API...'):
        try:
            response = requests.post(API_URL, json=input_data, timeout=10)
            if response.status_code == 200:
                prediction = response.json()['prediction']
                st.success(f"**Predicted Price: {prediction:,.2f} billion VND**")
            else:
                st.error("An error occurred with the API.")
                st.write(f"**Status Code:** `{response.status_code}`")
                st.json(response.json())
        except requests.exceptions.RequestException:
            st.error(f"**Connection Error:** Could not connect to the API.")
            st.warning(f"Please make sure the FastAPI server is running at `{API_URL}`.")