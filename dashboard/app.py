import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="PGCB Electricity Demand Forecasting | Bangladesh",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Mode Styling
st.markdown("""
<style>
    /* Global Container */
    .main {
        background-color: #0e1117;
    }
    
    /* Header Styling */
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #a0aec0;
        margin-bottom: 25px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #4facfe;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #cbd5e0;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        color: #cbd5e0;
        font-weight: 600;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0052d4 0%, #4364f7 50%, #6fb1fc 100%);
        color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Path Resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'PGCB_date_power_demand.xlsx')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, 'pgcb+hourly+generation+dataset+(bangladesh)', 'PGCB_date_power_demand.xlsx')
    
METRICS_PATH = os.path.join(BASE_DIR, 'outputs', 'metrics_comparison.csv')
PREDS_PATH = os.path.join(BASE_DIR, 'outputs', 'predictions_test.csv')

# Load Data Functions (Cached for Performance)
@st.cache_data
def load_raw_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_excel(DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    return None

@st.cache_data
def load_prediction_outputs():
    if os.path.exists(PREDS_PATH) and os.path.exists(METRICS_PATH):
        preds_df = pd.read_csv(PREDS_PATH)
        preds_df['datetime'] = pd.to_datetime(preds_df['datetime'])
        metrics_df = pd.read_csv(METRICS_PATH)
        return preds_df, metrics_df
    return None, None

raw_df = load_raw_dataset()
preds_df, metrics_df = load_prediction_outputs()

if raw_df is None or preds_df is None:
    st.error("Error: Required dataset or model predictions not found. Please ensure pipeline execution has completed.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/color/96/000000/lightning-bolt.png", width=64)
st.sidebar.title("⚡ Control Center")
st.sidebar.markdown("---")

# Model Selection
model_choice = st.sidebar.selectbox(
    "🎯 Select Forecasting Model",
    options=["LightGBM Regressor", "XGBoost Regressor", "PyTorch LSTM", "Compare All Models"],
    index=0
)

# Date Filter on Prediction Horizon
min_date = preds_df['datetime'].min().date()
max_date = preds_df['datetime'].max().date()

st.sidebar.markdown("### 📅 Time Window")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - pd.Timedelta(days=7), max_date),
    min_value=min_date,
    max_value=max_date
)

# Quick Horizon Buttons
st.sidebar.markdown("### ⏱️ Quick Horizon")
horizon = st.sidebar.radio(
    "Filter Horizon",
    options=["Custom Range", "Last 24 Hours", "Last 48 Hours", "Last 7 Days", "Full Test Set"],
    index=0
)

# Filter Data according to controls
if horizon == "Last 24 Hours":
    start_dt = preds_df['datetime'].max() - pd.Timedelta(hours=24)
    end_dt = preds_df['datetime'].max()
elif horizon == "Last 48 Hours":
    start_dt = preds_df['datetime'].max() - pd.Timedelta(hours=48)
    end_dt = preds_df['datetime'].max()
elif horizon == "Last 7 Days":
    start_dt = preds_df['datetime'].max() - pd.Timedelta(days=7)
    end_dt = preds_df['datetime'].max()
elif horizon == "Full Test Set":
    start_dt = preds_df['datetime'].min()
    end_dt = preds_df['datetime'].max()
else: # Custom Range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(hours=23, minutes=59)
    else:
        start_dt = preds_df['datetime'].min()
        end_dt = preds_df['datetime'].max()

filtered_preds = preds_df[(preds_df['datetime'] >= start_dt) & (preds_df['datetime'] <= end_dt)].copy()

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Context:**\n"
    "BEPRC Funded AI Research Project\n"
    "PGCB Electricity Demand Forecasting (Bangladesh)\n"
    "**Evaluated Models:** LightGBM, XGBoost, PyTorch LSTM"
)

# --- MAIN DASHBOARD HEADER ---
st.markdown('<div class="header-title">⚡ PGCB Electricity Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Real-time Historical Analytics & Machine Learning / Deep Learning Time-Series Predictions (Bangladesh Grid)</div>', unsafe_allow_html=True)

# --- KPI METRIC CARDS ROW ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    latest_val = filtered_preds['actual_demand_mw'].iloc[-1] if not filtered_preds.empty else 0
    prev_24h_val = filtered_preds['actual_demand_mw'].iloc[-25] if len(filtered_preds) >= 25 else latest_val
    delta_val = latest_val - prev_24h_val
    st.metric("Latest Demand", f"{latest_val:,.0f} MW", delta=f"{delta_val:,.0f} MW vs 24h ago")

with kpi_col2:
    peak_val = filtered_preds['actual_demand_mw'].max() if not filtered_preds.empty else 0
    st.metric("Peak Demand (Window)", f"{peak_val:,.0f} MW")

with kpi_col3:
    avg_val = filtered_preds['actual_demand_mw'].mean() if not filtered_preds.empty else 0
    st.metric("Average Load", f"{avg_val:,.0f} MW")

with kpi_col4:
    if model_choice != "Compare All Models":
        sel_row = metrics_df[metrics_df['Model'].str.contains(model_choice.split()[0], case=False)]
        if not sel_row.empty:
            mape_str = f"{sel_row['MAPE'].values[0]:.2f}%"
            mae_str = f"{sel_row['MAE'].values[0]:.1f} MW"
            st.metric(f"{model_choice} MAPE", mape_str, delta=f"MAE: {mae_str}", delta_color="normal")
        else:
            st.metric("Model Error", "N/A")
    else:
        best_row = metrics_df.loc[metrics_df['MAPE'].idxmin()]
        st.metric("Best Model (LightGBM)", f"{best_row['MAPE']:.2f}% MAPE", delta=f"MAE: {best_row['MAE']:.1f} MW", delta_color="normal")

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS CONTAINER ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Forecast Overlay", 
    "⚡ Generation & Fuel-Mix Analysis", 
    "📉 Model Performance & Residuals", 
    "📥 Data Table & Export"
])

# --- TAB 1: FORECAST OVERLAY ---
with tab1:
    st.subheader("Historical Demand vs. AI Model Predictions")
    
    fig = go.Figure()
    
    # Actual Demand Line
    fig.add_trace(go.Scatter(
        x=filtered_preds['datetime'],
        y=filtered_preds['actual_demand_mw'],
        mode='lines',
        name='Actual Demand (MW)',
        line=dict(color='#00f2fe', width=3)
    ))
    
    # Model Lines according to selection
    if model_choice in ["LightGBM Regressor", "Compare All Models"]:
        fig.add_trace(go.Scatter(
            x=filtered_preds['datetime'],
            y=filtered_preds['lightgbm_pred_mw'],
            mode='lines',
            name='LightGBM Forecast (1.79% MAPE)',
            line=dict(color='#2ca02c', width=2, dash='dash')
        ))
        
    if model_choice in ["XGBoost Regressor", "Compare All Models"]:
        fig.add_trace(go.Scatter(
            x=filtered_preds['datetime'],
            y=filtered_preds['xgboost_pred_mw'],
            mode='lines',
            name='XGBoost Forecast (1.83% MAPE)',
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ))
        
    if model_choice in ["PyTorch LSTM", "Compare All Models"]:
        fig.add_trace(go.Scatter(
            x=filtered_preds['datetime'],
            y=filtered_preds['lstm_pred_mw'],
            mode='lines',
            name='PyTorch LSTM Forecast (2.37% MAPE)',
            line=dict(color='#e377c2', width=2, dash='dashdot')
        ))
        
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Datetime", showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(title="Electricity Demand (MW)", showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: FUEL-MIX & GENERATION ---
with tab2:
    st.subheader("Bangladesh Energy Source Breakdown & Seasonal Analysis")
    
    col_fuel1, col_fuel2 = st.columns(2)
    
    fuel_cols = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind']
    avail_fuels = [c for c in fuel_cols if c in raw_df.columns]
    
    with col_fuel1:
        if avail_fuels:
            fuel_sum = raw_df[avail_fuels].mean().reset_index()
            fuel_sum.columns = ['Fuel Source', 'Average MW']
            fuel_sum['Fuel Source'] = fuel_sum['Fuel Source'].str.replace('_', ' ').str.title()
            
            fig_pie = px.pie(
                fuel_sum, 
                values='Average MW', 
                names='Fuel Source',
                title="Historical Fuel-Mix Share (2015-2025 Average)",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col_fuel2:
        # Diurnal Hourly Demand Profile
        raw_df['hour'] = raw_df['datetime'].dt.hour
        hourly_profile = raw_df.groupby('hour')['demand_mw'].agg(['mean', 'max', 'min']).reset_index()
        
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(x=hourly_profile['hour'], y=hourly_profile['mean'], name='Average Demand', line=dict(color='#4facfe', width=3)))
        fig_profile.add_trace(go.Scatter(x=hourly_profile['hour'], y=hourly_profile['max'], name='Peak Demand', line=dict(color='#ff4b4b', width=1.5, dash='dash')))
        fig_profile.add_trace(go.Scatter(x=hourly_profile['hour'], y=hourly_profile['min'], name='Min Demand', line=dict(color='#00f2fe', width=1.5, dash='dot')))
        
        fig_profile.update_layout(
            title="24-Hour Diurnal Demand Pattern (MW)",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Hour of Day (0-23)", tickmode='linear', tick0=0, dtick=2)
        )
        st.plotly_chart(fig_profile, use_container_width=True)

# --- TAB 3: MODEL ACCURACY & RESIDUALS ---
with tab3:
    st.subheader("Quantitative Model Metrics & Error Distributions")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Model Metrics Comparison (MAE, RMSE, MAPE)")
        st.dataframe(
            metrics_df.style.format({
                'MAE': '{:.2f} MW',
                'RMSE': '{:.2f} MW',
                'MAPE': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Bar Chart
        fig_metrics = px.bar(
            metrics_df,
            x='Model',
            y='MAPE',
            text='MAPE',
            title='MAPE (%) Comparison Across Models (Lower is Better)',
            color='Model',
            color_discrete_sequence=['#2ca02c', '#1f77b4', '#d62728']
        )
        fig_metrics.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_metrics.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    with col_m2:
        st.markdown("#### Forecast Residual Error Distribution")
        res_lgb = filtered_preds['actual_demand_mw'] - filtered_preds['lightgbm_pred_mw']
        res_xgb = filtered_preds['actual_demand_mw'] - filtered_preds['xgboost_pred_mw']
        res_lstm = filtered_preds['actual_demand_mw'] - filtered_preds['lstm_pred_mw']
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=res_lgb, name='LightGBM Residuals', opacity=0.6, marker_color='#2ca02c'))
        fig_res.add_trace(go.Histogram(x=res_xgb, name='XGBoost Residuals', opacity=0.6, marker_color='#ff7f0e'))
        fig_res.add_trace(go.Histogram(x=res_lstm, name='LSTM Residuals', opacity=0.6, marker_color='#e377c2'))
        
        fig_res.update_layout(
            title="Residual Error Distribution (Actual - Predicted MW)",
            barmode='overlay',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_res, use_container_width=True)

# --- TAB 4: DATA EXPLORER & EXPORT ---
with tab4:
    st.subheader("Filtered Predictions Data Explorer & One-Click Download")
    
    st.dataframe(filtered_preds.style.format({
        'actual_demand_mw': '{:,.1f}',
        'lightgbm_pred_mw': '{:,.1f}',
        'xgboost_pred_mw': '{:,.1f}',
        'lstm_pred_mw': '{:,.1f}'
    }), height=400, use_container_width=True)
    
    csv_data = filtered_preds.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Predictions to CSV",
        data=csv_data,
        file_name="pgcb_electricity_forecast_predictions.csv",
        mime="text/csv",
        help="Download filtered actual vs predicted values dataframe as a CSV file."
    )

# Footer
st.markdown("""
<div class="footer">
    PGCB Hourly Electricity Demand Forecasting System | Built for BEPRC Funded AI Research Project Submission<br>
    Developed with Python, Streamlit, Plotly, LightGBM, XGBoost, and PyTorch.
</div>
""", unsafe_allow_html=True)
