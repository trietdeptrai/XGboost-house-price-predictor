from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np

model = joblib.load("xgboost_original.pkl")
preprocessor = joblib.load("preprocessor.pkl")

class InputData(BaseModel):
    Area: float
    Floors: int
    Bedrooms: int
    Bathrooms: int
    Frontage: float
    Access_road: float = Field(alias="Access road")   
    Legal_status: str = Field(alias="Legal status") 

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # 1. Create a DataFrame from the raw input data
    # Use by_alias=True to get columns like "Access road" with a space
    df = pd.DataFrame([data.model_dump(by_alias=True)])

    # 2. --- APPLY FEATURE ENGINEERING ---
    
    
    # Create a copy to avoid SettingWithCopyWarning
    df_engineered = df.copy()

    # Apply log transformations. Using np.log1p is safer as it handles zeros (log1p(0) = 0).
    df_engineered['log_Area'] = np.log1p(df_engineered['Area'])
    df_engineered['log_Frontage'] = np.log1p(df_engineered['Frontage'])
    df_engineered['log_access_road'] = np.log1p(df_engineered['Access road'])

    # Create the "missing" flags.
    # (This is an assumption on how you created them. If you used a different rule, adjust it here.)
    # For example, if a value of 0 meant it was missing:
    df_engineered['Frontage_missing'] = (df_engineered['Frontage'] == 0).astype(int)
    df_engineered['AccessRoad_missing'] = (df_engineered['Access road'] == 0).astype(int)

    # 3. --- PREPARE DATA FOR THE PREPROCESSOR ---
    # The preprocessor expects the DataFrame to have specific columns. Let's ensure it gets them.
    # Based on your training code, these are the columns it needs.
    numeric_features = ['log_Area', 'Floors', 'Bedrooms', 'Bathrooms', 'Frontage_missing', 'log_Frontage', 'log_access_road', 'AccessRoad_missing']
    categorical_features = ['Legal status']
    
    # Combine the lists to define the final feature set for the model
    final_features = categorical_features + numeric_features

    # 4. Transform the data using the preprocessor
    # We pass the engineered dataframe with the correct columns
    transformed = preprocessor.transform(df_engineered[final_features])

    # 5. Make a prediction
    prediction = model.predict(transformed)

    # 6. Return the result
    return {"prediction": float(prediction[0])}