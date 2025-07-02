import pandas as pd
import numpy as np

def load_and_engineer_features(file_path: str):
    """
    Loads raw data, performs feature engineering, and basic cleaning.

    Args:
        file_path (str): Path to the raw CSV data file.

    Returns:
        pd.DataFrame: A DataFrame with new features and ready for preprocessing.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # --- Feature Engineering from your notebook ---
    print("Engineering new features...")
    # 1. Log transformations
    df['log_Area'] = np.log1p(df['Area'])
    df['log_Frontage'] = np.log1p(df['Frontage'])
    df['log_access_road'] = np.log1p(df['Access Road'])

    # 2. Missing value flags
    df['Frontage_missing'] = df['Frontage'].isna().astype(int)
    df['AccessRoad_missing'] = df['Access Road'].isna().astype(int)

    # 3. Ratio feature
    df['bedrooms_per_bathroom'] = df['Bedrooms'] / df['Bathrooms'].replace(0, np.nan)

    # --- Basic Cleaning from your notebook ---
    print("Selecting and cleaning final columns...")
    # Select only the columns needed for the model and drop rows with missing target or crucial features
    selected_columns = [
        'log_Area', 'Floors', 'Bedrooms', 'Bathrooms', 'Legal status', 
        'Price', 'Frontage_missing', 'log_Frontage', 'log_access_road', 
        'AccessRoad_missing', 'bedrooms_per_bathroom'
    ]
    # Keep only columns that exist in the dataframe to avoid errors
    final_columns = [col for col in selected_columns if col in df.columns]
    
    clean_df = df[final_columns].copy()
    
    # Drop rows where the target 'Price' is missing
    clean_df.dropna(subset=['Price'], inplace=True)
    
    print("Data loading and feature engineering complete.")
    return clean_df