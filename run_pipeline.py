import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src path
base_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

from preprocessing import load_and_preprocess_data, get_train_test_split
from train_ml import train_xgboost, train_lightgbm
from train_lstm import train_pytorch_lstm

def main():
    excel_path = os.path.join(base_dir, 'pgcb+hourly+generation+dataset+(bangladesh)', 'PGCB_date_power_demand.xlsx')
    if not os.path.exists(excel_path):
        excel_path = os.path.join(base_dir, 'data', 'PGCB_date_power_demand.xlsx')
        
    models_dir = os.path.join(base_dir, 'models')
    outputs_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(outputs_dir, 'figures')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=== Phase 1: Loading & Preprocessing Dataset ===")
    df = load_and_preprocess_data(excel_path)
    print(f"Data shape after feature engineering: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Train / Test split (80% / 20%)
    train_df, test_df = get_train_test_split(df, test_size=0.2)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    print("\n=== Phase 2: Exploratory Data Analysis & Saving Figures ===")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Hourly Demand Profile
    plt.figure(figsize=(10, 5))
    hourly_avg = df.groupby('hour')['demand_mw'].mean()
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o', color='#1f77b4', linewidth=2.5)
    plt.title('Average Electricity Demand by Hour of Day (Bangladesh PGCB)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Demand (MW)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '1_hourly_demand_profile.png'), dpi=300)
    plt.close()
    
    # 2. Monthly Trend & Seasonality
    plt.figure(figsize=(12, 5))
    monthly_avg = df.groupby(['year', 'month'])['demand_mw'].mean().reset_index()
    monthly_avg['year_month'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str).str.zfill(2)
    sns.barplot(data=monthly_avg.tail(36), x='year_month', y='demand_mw', palette='viridis')
    plt.title('Monthly Average Electricity Demand (Recent 3 Years)', fontsize=14, fontweight='bold')
    plt.xlabel('Year-Month', fontsize=10)
    plt.ylabel('Average Demand (MW)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '2_monthly_demand_trend.png'), dpi=300)
    plt.close()
    
    # 3. Fuel-Mix Generation Breakdown
    fuel_cols = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind']
    avail_fuel = [c for c in fuel_cols if c in df.columns]
    if avail_fuel:
        plt.figure(figsize=(8, 8))
        fuel_means = df[avail_fuel].mean()
        plt.pie(fuel_means, labels=[c.replace('_', ' ').title() for c in avail_fuel], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('tab10'))
        plt.title('Bangladesh Generation Fuel-Mix Breakdown (2015-2025 Average)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, '3_fuel_mix_breakdown.png'), dpi=300)
        plt.close()
        
    print("EDA visual figures saved to outputs/figures/")
    
    print("\n=== Phase 3: Model Training & Evaluation ===")
    
    # XGBoost
    print("Training XGBoost Regressor...")
    xgb_model_path = os.path.join(models_dir, 'xgboost_model.json')
    xgb_model, xgb_preds, xgb_metrics = train_xgboost(train_df, test_df, xgb_model_path)
    print(f"XGBoost Results -> MAE: {xgb_metrics['MAE']:.2f} MW, RMSE: {xgb_metrics['RMSE']:.2f} MW, MAPE: {xgb_metrics['MAPE']:.2f}%")
    
    # LightGBM
    print("Training LightGBM Regressor...")
    lgb_model_path = os.path.join(models_dir, 'lightgbm_model.txt')
    lgb_model, lgb_preds, lgb_metrics = train_lightgbm(train_df, test_df, lgb_model_path)
    print(f"LightGBM Results -> MAE: {lgb_metrics['MAE']:.2f} MW, RMSE: {lgb_metrics['RMSE']:.2f} MW, MAPE: {lgb_metrics['MAPE']:.2f}%")
    
    # PyTorch LSTM
    print("Training PyTorch LSTM Regressor...")
    lstm_model_path = os.path.join(models_dir, 'lstm_model.pt')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    seq_length = 24
    lstm_model, lstm_preds, lstm_actuals, lstm_metrics = train_pytorch_lstm(
        train_df, test_df, lstm_model_path, scaler_path, seq_length=seq_length, epochs=15, batch_size=128
    )
    print(f"PyTorch LSTM Results -> MAE: {lstm_metrics['MAE']:.2f} MW, RMSE: {lstm_metrics['RMSE']:.2f} MW, MAPE: {lstm_metrics['MAPE']:.2f}%")
    
    # Metrics Comparison Table
    metrics_summary = pd.DataFrame([
        {'Model': 'XGBoost Regressor', **xgb_metrics},
        {'Model': 'LightGBM Regressor', **lgb_metrics},
        {'Model': 'PyTorch LSTM', **lstm_metrics}
    ])
    metrics_csv_path = os.path.join(outputs_dir, 'metrics_comparison.csv')
    metrics_summary.to_csv(metrics_csv_path, index=False)
    print("\nMetrics Summary Table:")
    print(metrics_summary.to_string(index=False))
    
    # Save Predictions CSV
    test_sub = test_df.iloc[seq_length:].copy().reset_index(drop=True)
    preds_df = pd.DataFrame({
        'datetime': test_sub['datetime'],
        'actual_demand_mw': lstm_actuals,
        'xgboost_pred_mw': xgb_preds[seq_length:],
        'lightgbm_pred_mw': lgb_preds[seq_length:],
        'lstm_pred_mw': lstm_preds
    })
    preds_csv_path = os.path.join(outputs_dir, 'predictions_test.csv')
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"\nTest predictions saved to {preds_csv_path}")
    
    # 4. Actual vs Predicted Comparison Plot (Last 168 hours = 1 week sample)
    plt.figure(figsize=(14, 6))
    sample_df = preds_df.tail(168).reset_index(drop=True)
    plt.plot(sample_df['datetime'], sample_df['actual_demand_mw'], label='Actual Demand', color='black', linewidth=2.5)
    plt.plot(sample_df['datetime'], sample_df['xgboost_pred_mw'], label='XGBoost Forecast', color='#1f77b4', linestyle='--')
    plt.plot(sample_df['datetime'], sample_df['lightgbm_pred_mw'], label='LightGBM Forecast', color='#2ca02c', linestyle='-.')
    plt.plot(sample_df['datetime'], sample_df['lstm_pred_mw'], label='PyTorch LSTM Forecast', color='#d62728', linestyle=':')
    plt.title('Electricity Demand Forecast Comparison (1-Week Sample Test Horizon)', fontsize=14, fontweight='bold')
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel('Demand (MW)', fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '4_forecast_comparison.png'), dpi=300)
    plt.close()
    
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == '__main__':
    main()
