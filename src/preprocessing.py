import os
import pandas as pd
import numpy as np

def load_and_preprocess_data(excel_path):
    """
    Loads PGCB Hourly Generation dataset, handles datetime conversion,
    feature engineering (calendar, cyclical, lags, rolling stats).
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Dataset not found at {excel_path}")
    
    df = pd.read_excel(excel_path)
    
    # Ensure datetime sorting
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Fill any minor missing numerical values with interpolation
    num_cols = ['demand_mw', 'generation_mw', 'load_shedding', 'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear').ffill().bfill()
            
    # Feature Engineering - Calendar Features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['is_weekend'] = df['dayofweek'].isin([4, 5]).astype(int) # Friday, Saturday in Bangladesh
    
    # Cyclical Features (Sine/Cosine for hour and month)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Target variable: Electricity Demand in MW
    target = 'demand_mw'
    
    # Lag Features
    df['demand_lag_1'] = df[target].shift(1)
    df['demand_lag_24'] = df[target].shift(24)
    df['demand_lag_168'] = df[target].shift(168)
    
    # Rolling Window Features
    df['demand_roll_mean_24'] = df[target].shift(1).rolling(window=24).mean()
    df['demand_roll_std_24'] = df[target].shift(1).rolling(window=24).std()
    df['demand_roll_mean_168'] = df[target].shift(1).rolling(window=168).mean()
    df['demand_roll_std_168'] = df[target].shift(1).rolling(window=168).std()
    
    # Drop initial NaN rows resulting from 168h lags
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def get_train_test_split(df, test_size=0.2):
    """
    Time-series train/test split with zero shuffling.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df
