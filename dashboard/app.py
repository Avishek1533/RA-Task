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

# Custom Vibrant & Professional Theme Styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background Styling */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1322 100%);
        color: #f3f4f6;
    }
    
    /* Title Header Styling */
    .header-box {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 6px solid #38bdf8;
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(14, 165, 233, 0.25);
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        margin-left: 10px;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 3px solid #38bdf8;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-top-color: #c084fc;
        box-shadow: 0 12px 30px rgba(192, 132, 252, 0.25);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-size: 0.88rem;
        font-weight: 600;
    }
    
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 800;
        font-size: 1.85rem;
        letter-spacing: -0.5px;
    }

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background-color: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 22px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background: rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #090d16 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: all 0.2s ease;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #34d399 0%, #10b981 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        transform: translateY(-2px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Path Resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OPTIMIZED HIGH-SPEED FILE PATHS (Parquet > CSV > Excel)
RAW_PARQUET = os.path.join(BASE_DIR, 'data', 'PGCB_date_power_demand.parquet')
RAW_EXCEL = os.path.join(BASE_DIR, 'data', 'PGCB_date_power_demand.xlsx')
if not os.path.exists(RAW_EXCEL):
    RAW_EXCEL = os.path.join(BASE_DIR, 'pgcb+hourly+generation+dataset+(bangladesh)', 'PGCB_date_power_demand.xlsx')

PREDS_PARQUET = os.path.join(BASE_DIR, 'outputs', 'predictions_test.parquet')
PREDS_CSV = os.path.join(BASE_DIR, 'outputs', 'predictions_test.csv')
METRICS_CSV = os.path.join(BASE_DIR, 'outputs', 'metrics_comparison.csv')

# --- CACHED HIGH-SPEED DATA LOADERS (Sub-second Parquet loading) ---
@st.cache_data(show_spinner=False, ttl=3600)
def load_raw_dataset():
    if os.path.exists(RAW_PARQUET):
        df = pd.read_parquet(RAW_PARQUET, engine='pyarrow')
        return df
    elif os.path.exists(RAW_EXCEL):
        df = pd.read_excel(RAW_EXCEL)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_prediction_outputs():
    preds_df = None
    if os.path.exists(PREDS_PARQUET):
        preds_df = pd.read_parquet(PREDS_PARQUET, engine='pyarrow')
    elif os.path.exists(PREDS_CSV):
        preds_df = pd.read_csv(PREDS_CSV)
        preds_df['datetime'] = pd.to_datetime(preds_df['datetime'])
        
    metrics_df = None
    if os.path.exists(METRICS_CSV):
        metrics_df = pd.read_csv(METRICS_CSV)
        
    return preds_df, metrics_df

# Load datasets with spinner
with st.spinner("⚡ Loading high-speed time-series models & dataset..."):
    raw_df = load_raw_dataset()
    preds_df, metrics_df = load_prediction_outputs()

if raw_df is None or preds_df is None:
    st.error("Error: Required dataset or model predictions not found. Please ensure pipeline execution has completed.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown(
    """
    <div style="text-align: left; padding-bottom: 10px;">
        <img src="https://img.icons8.com/fluency/96/lightning-bolt.png" width="65" style="filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.8));">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("<h2 style='color:#38bdf8; font-weight:800; margin-bottom:0;'>⚡ Control Center</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>Interactive Filter & Model Selector</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Model Selection
model_choice = st.sidebar.selectbox(
    "🎯 Select Forecasting Model",
    options=["LightGBM Regressor", "XGBoost Regressor", "PyTorch LSTM", "Compare All Models"],
    index=0
)

# Date Filter
min_date = preds_df['datetime'].min().date()
max_date = preds_df['datetime'].max().date()

st.sidebar.markdown("### 📅 Date Range Filter")
date_range = st.sidebar.date_input(
    "Custom Range Selector",
    value=(max_date - pd.Timedelta(days=7), max_date),
    min_value=min_date,
    max_value=max_date
)

# Quick Horizon Buttons
st.sidebar.markdown("### ⏱️ Preset Horizon")
horizon = st.sidebar.radio(
    "Select Window",
    options=["Custom Range", "Last 24 Hours", "Last 48 Hours", "Last 7 Days", "Full Test Set"],
    index=0
)

# Filter Logic
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
else:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(hours=23, minutes=59)
    else:
        start_dt = preds_df['datetime'].min()
        end_dt = preds_df['datetime'].max()

filtered_preds = preds_df[(preds_df['datetime'] >= start_dt) & (preds_df['datetime'] <= end_dt)].copy()

# Sidebar Metadata Box
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background: rgba(30, 41, 59, 0.6); padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.82rem; color: #cbd5e1;'>
        <b style='color:#38bdf8;'>BEPRC Research Project</b><br>
        PGCB Electricity Demand Forecasting<br>
        <span style='color:#94a3b8;'>Bangladesh Power Grid (2015–2025)</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- MAIN DASHBOARD HEADER ---
st.markdown(
    """
    <div class="header-box">
        <div class="header-title">⚡ PGCB Electricity Demand Forecasting Dashboard</div>
        <div class="header-subtitle">Real-Time Historical Analytics & Machine Learning / Deep Learning Time-Series Predictions (Bangladesh Grid)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- VIBRANT KPI METRIC CARDS ROW ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    latest_val = filtered_preds['actual_demand_mw'].iloc[-1] if not filtered_preds.empty else 0
    prev_24h_val = filtered_preds['actual_demand_mw'].iloc[-25] if len(filtered_preds) >= 25 else latest_val
    delta_val = latest_val - prev_24h_val
    st.metric("Latest Demand", f"{latest_val:,.0f} MW", delta=f"{delta_val:+,.0f} MW vs 24h ago")

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
    "📈 Forecast Overlay & Comparison", 
    "⚡ Fuel-Mix & Generation Breakdown", 
    "📉 Model Performance & Residuals", 
    "📥 Data Explorer & Export"
])

# --- TAB 1: FORECAST OVERLAY ---
with tab1:
    st.markdown("<h3 style='color:#f8fafc; font-weight:700;'>Historical Electricity Demand vs. Model Forecasts</h3>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Actual Demand Line (Glowing Neon Cyan)
    fig.add_trace(go.Scatter(
        x=filtered_preds['datetime'],
        y=filtered_preds['actual_demand_mw'],
        mode='lines',
        name='Actual Demand (MW)',
        line=dict(color='#00f2fe', width=3.5)
    ))
    
    # Model Lines with Vibrant Neon Colors
    if model_choice in ["LightGBM Regressor", "Compare All Models"]:
        fig.add_trace(go.Scatter(
            x=filtered_preds['datetime'],
            y=filtered_preds['lightgbm_pred_mw'],
            mode='lines',
            name='LightGBM Forecast (1.79% MAPE)',
            line=dict(color='#00e676', width=2.5, dash='dash')
        ))
        
    if model_choice in ["XGBoost Regressor", "Compare All Models"]:
        fig.add_trace(go.Scatter(
            x=filtered_preds['datetime'],
            y=filtered_preds['xgboost_pred_mw'],
            mode='lines',
            name='XGBoost Forecast (1.83% MAPE)',
            line=dict(color='#ff9100', width=2.5, dash='dot')
        ))
        
    if model_choice in ["PyTorch LSTM", "Compare All Models"]:
        fig.add_trace(go.Scatter(
            x=filtered_preds['datetime'],
            y=filtered_preds['lstm_pred_mw'],
            mode='lines',
            name='PyTorch LSTM Forecast (2.37% MAPE)',
            line=dict(color='#f50057', width=2.5, dash='dashdot')
        ))
        
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=540,
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        xaxis=dict(title="Datetime", showgrid=True, gridcolor='rgba(255,255,255,0.07)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Electricity Demand (MW)", showgrid=True, gridcolor='rgba(255,255,255,0.07)', zerolinecolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12))
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: FUEL-MIX & GENERATION ---
with tab2:
    st.markdown("<h3 style='color:#f8fafc; font-weight:700;'>Bangladesh Energy Source Share & Diurnal Patterns</h3>", unsafe_allow_html=True)
    
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
                title="<b>Historical Fuel-Mix Share (2015-2025 Average)</b>",
                hole=0.45,
                color_discrete_sequence=['#38bdf8', '#fbbf24', '#f87171', '#34d399', '#c084fc', '#a7f3d0']
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
            fig_pie.update_layout(
                template='plotly_dark', 
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col_fuel2:
        if 'hour' not in raw_df.columns:
            raw_df['hour'] = raw_df['datetime'].dt.hour
            
        hourly_profile = raw_df.groupby('hour')['demand_mw'].agg(['mean', 'max', 'min']).reset_index()
        
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(x=hourly_profile['hour'], y=hourly_profile['mean'], name='Average Demand', line=dict(color='#38bdf8', width=3.5)))
        fig_profile.add_trace(go.Scatter(x=hourly_profile['hour'], y=hourly_profile['max'], name='Peak Demand', line=dict(color='#f43f5e', width=2, dash='dash')))
        fig_profile.add_trace(go.Scatter(x=hourly_profile['hour'], y=hourly_profile['min'], name='Min Demand', line=dict(color='#34d399', width=2, dash='dot')))
        
        fig_profile.update_layout(
            title="<b>24-Hour Diurnal Demand Pattern (MW)</b>",
            template='plotly_dark',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            xaxis=dict(title="Hour of Day (0-23)", tickmode='linear', tick0=0, dtick=2, gridcolor='rgba(255,255,255,0.07)'),
            yaxis=dict(title="Demand (MW)", gridcolor='rgba(255,255,255,0.07)')
        )
        st.plotly_chart(fig_profile, use_container_width=True)

# --- TAB 3: MODEL ACCURACY & RESIDUALS ---
with tab3:
    st.markdown("<h3 style='color:#f8fafc; font-weight:700;'>Quantitative Performance Benchmark & Residual Analysis</h3>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Evaluation Metrics Table (Unseen Test Set)")
        st.dataframe(
            metrics_df.style.format({
                'MAE': '{:.2f} MW',
                'RMSE': '{:.2f} MW',
                'MAPE': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Bar Chart with Custom Colors
        fig_metrics = px.bar(
            metrics_df,
            x='Model',
            y='MAPE',
            text='MAPE',
            title='<b>MAPE (%) Comparison Across Models (Lower is Better)</b>',
            color='Model',
            color_discrete_sequence=['#00e676', '#ff9100', '#f50057']
        )
        fig_metrics.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_metrics.update_layout(template='plotly_dark', paper_bgcolor='rgba(15, 23, 42, 0.6)', plot_bgcolor='rgba(15, 23, 42, 0.6)')
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    with col_m2:
        st.markdown("#### Forecast Residual Error Distribution")
        res_lgb = filtered_preds['actual_demand_mw'] - filtered_preds['lightgbm_pred_mw']
        res_xgb = filtered_preds['actual_demand_mw'] - filtered_preds['xgboost_pred_mw']
        res_lstm = filtered_preds['actual_demand_mw'] - filtered_preds['lstm_pred_mw']
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=res_lgb, name='LightGBM Residuals', opacity=0.65, marker_color='#00e676'))
        fig_res.add_trace(go.Histogram(x=res_xgb, name='XGBoost Residuals', opacity=0.65, marker_color='#ff9100'))
        fig_res.add_trace(go.Histogram(x=res_lstm, name='LSTM Residuals', opacity=0.65, marker_color='#f50057'))
        
        fig_res.update_layout(
            title="<b>Residual Error Distribution (Actual - Predicted MW)</b>",
            barmode='overlay',
            template='plotly_dark',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)'
        )
        st.plotly_chart(fig_res, use_container_width=True)

# --- TAB 4: DATA EXPLORER & EXPORT ---
with tab4:
    st.markdown("<h3 style='color:#f8fafc; font-weight:700;'>Filtered Predictions Data Table & Export</h3>", unsafe_allow_html=True)
    
    st.dataframe(filtered_preds.style.format({
        'actual_demand_mw': '{:,.1f}',
        'lightgbm_pred_mw': '{:,.1f}',
        'xgboost_pred_mw': '{:,.1f}',
        'lstm_pred_mw': '{:,.1f}'
    }), height=380, use_container_width=True)
    
    csv_data = filtered_preds.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⚡ Export Filtered Predictions to CSV",
        data=csv_data,
        file_name="pgcb_electricity_forecast_predictions.csv",
        mime="text/csv",
        help="Click to download the current filtered dataframe as a CSV file."
    )

# Footer
st.markdown("""
<div class="footer">
    PGCB Hourly Electricity Demand Forecasting System | Built for BEPRC Funded AI Research Project Submission<br>
    Developed with Python, Streamlit, Plotly, LightGBM, XGBoost, PyTorch, and PyArrow Parquet.
</div>
""", unsafe_allow_html=True)
