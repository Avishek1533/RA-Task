import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

FEATURE_COLS = [
    'hour', 'dayofweek', 'day', 'month', 'year', 'is_weekend',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
    'demand_roll_mean_24', 'demand_roll_std_24',
    'demand_roll_mean_168', 'demand_roll_std_168'
]

TARGET_COL = 'demand_mw'

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def train_xgboost(train_df, test_df, model_save_path):
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test.values, preds)
    
    # Save model
    model.save_model(model_save_path)
    return model, preds, metrics

def train_lightgbm(train_df, test_df, model_save_path):
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test.values, preds)
    
    # Save model
    model.booster_.save_model(model_save_path)
    return model, preds, metrics
