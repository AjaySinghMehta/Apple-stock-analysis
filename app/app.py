"""
Interactive Stock Analysis App
- Runs locally with SQL Server (pyodbc available)
- Runs on Streamlit Cloud using automatic yfinance fallback
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

# =============================================================================
# FIX 1: Correct project root so local `src/db.py` loads in both local & cloud
# =============================================================================
project_root = Path(__file__).resolve().parents[0]  # <-- Correct
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# =============================================================================
# FIX 2: Auto-disable local DB if running on Streamlit Cloud
# =============================================================================
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS", "false") == "true"
USE_LOCAL_DB_DEFAULT = False if IS_CLOUD else True  # YOU can override

# =============================================================================
# Lazy loader for SQL Server
# =============================================================================
def _try_get_engine():
    try:
        from src.db import get_engine as _get
        return _get
    except Exception:
        return None


# =============================================================================
# Safe unified fetch: SQL Server (local) OR yfinance (cloud)
# =============================================================================
@st.cache_data(ttl=300)
def fetch_stock_from_sql(ticker: str, start: str, end: str, use_local: bool = False) -> pd.DataFrame:
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end   = pd.to_datetime(end).strftime("%Y-%m-%d")

    # ---------------------- SQL PATH (LOCAL ONLY) ----------------------
    if use_local and not IS_CLOUD:
        get_engine = _try_get_engine()
        if get_engine:
            try:
                engine = get_engine()
                query = text("""
                    SELECT [stock_date], [ticker], [open], [high], [low], [close], [adj_close], [volume]
                    FROM dbo.stock_daily
                    WHERE ticker = :ticker AND stock_date BETWEEN :start AND :end
                    ORDER BY [stock_date] ASC
                """)
                with engine.connect() as conn:
                    df = pd.read_sql(query, conn, params={"ticker": ticker, "start": start, "end": end})

                df["stock_date"] = pd.to_datetime(df["stock_date"])
                numeric_cols = ["open","high","low","close","adj_close","volume"]
                for c in numeric_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                return df.sort_values("stock_date").reset_index(drop=True)

            except Exception as e:
                st.warning("⚠️ Local SQL query failed. Falling back to yfinance.")

        else:
            st.info("ℹ️ Local SQL Server unavailable. Using yfinance.")

    # ---------------------- CLOUD/YFINANCE PATH ----------------------
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)

        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.reset_index().rename(columns={
                "Date": "stock_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume"
            })
            df["ticker"] = ticker
            df["stock_date"] = pd.to_datetime(df["stock_date"])

            return df.sort_values("stock_date").reset_index(drop=True)

    except Exception as e:
        st.error(f"yfinance error: {e}")

    cols = ["stock_date","ticker","open","high","low","close","adj_close","volume"]
    return pd.DataFrame(columns=cols)


# =============================================================================
# Analysis helpers
# =============================================================================
def add_returns_and_mas(df):
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["cumulative"] = (1 + df["daily_return"]).cumprod() - 1
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA100"] = df["close"].rolling(100).mean()
    return df

def calc_yearly_cum_returns(df):
    res = {}
    for y in range(1, 6):
        days = int(252 * y)
        sub = df.tail(days)
        if len(sub) > 1:
            res[f"{y}Y"] = float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1)
        else:
            res[f"{y}Y"] = None
    return res

def monthly_avg_returns(df):
    df = df.copy()
    df["year"] = df["stock_date"].dt.year
    df["month"] = df["stock_date"].dt.month
    df["daily_return"] = df["close"].pct_change()
    out = df.groupby(["year","month"])["daily_return"].mean().reset_index()
    out["period"] = out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2)
    return out[["period","daily_return"]]


# =============================================================================
# Linear forecast
# =============================================================================
def linear_forecast(df, days_forecast=30, fit_days=120):
    try:
        df2 = df.dropna(subset=["close"]).sort_values("stock_date")
        if len(df2) < 10:
            return None

        fit_days = min(max(10, fit_days), len(df2))
        fit = df2.tail(fit_days).reset_index(drop=True)

        x = np.arange(len(fit))
        y = fit["close"].to_numpy()

        a, b = np.polyfit(x, y, 1)

        last_date = pd.to_datetime(fit["stock_date"].iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days_forecast)

        future_x = np.arange(len(fit), len(fit) + len(future_dates))
        forecast = a * future_x + b

        return pd.DataFrame({"stock_date": future_dates, "forecast_close": forecast})

    except:
        return None


# =============================================================================
# PLOTS
# =============================================================================
def plot_price(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["stock_date"], y=df["close"], name="Close", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["stock_date"], y=df["open"], name="Open", line=dict(width=1, dash="dash")))
    return fig

def plot_volume(df):
    return px.bar(df, x="stock_date", y="volume")

def plot_mas(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["stock_date"], y=df["close"], name="Close", opacity=0.6))
    for ma in ["MA20","MA50","MA100"]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df["stock_date"], y=df[ma], name=ma))
    return fig

def plot_cumulative(df):
    df = df.sort_values("stock_date").copy()
    df["daily_return"] = df["close"].pct_change()
    df["cumulative"] = (1 + df["daily_return"]).cumprod() - 1
    return px.line(df, x="stock_date", y="cumulative", title="Cumulative Return")


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.title("📈 Interactive Stock Analytics (SQL + Streamlit + Plotly)")

# Sidebar
st.sidebar.header("Query Options")

ticker = st.sidebar.selectbox("Ticker:", ["AAPL"])

# Quick range picker
quick = st.sidebar.selectbox("Range", ["5y","3y","2y","1y","6m","1m","custom"])

now = pd.Timestamp.now().normalize()

if quick == "custom":
    start_date = st.sidebar.date_input("Start Date", now - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", now)
else:
    years = {"5y":5,"3y":3,"2y":2,"1y":1}
    months = {"6m":6,"1m":1}
    if quick in years:
        start_date = now - pd.DateOffset(years=years[quick])
    else:
        start_date = now - pd.DateOffset(months=months[quick])
    end_date = now

start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

# FIX 3: Allow user to toggle SQL Server locally
use_local = st.sidebar.checkbox("Use Local SQL Server", value=USE_LOCAL_DB_DEFAULT)

st.sidebar.write(f"Fetching **{ticker}** from **{start_date} → {end_date}**")

# FETCH BUTTON
if st.sidebar.button("Fetch & Analyze"):
    with st.spinner("Fetching data..."):
        df = fetch_stock_from_sql(ticker, start_date, end_date, use_local=use_local)

    if df.empty:
        st.error("No data found.")
        st.stop()

    # KPIs
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Start Date", df["stock_date"].min().strftime("%Y-%m-%d"))
    col2.metric("End Date", df["stock_date"].max().strftime("%Y-%m-%d"))
    col3.metric("Start Price", f"${df['close'].iloc[0]:,.2f}")
    col4.metric("Latest Price", f"${df['close'].iloc[-1]:,.2f}")

    # Tabs
    tabs = st.tabs(["Price Trend","Volume","Moving Avg","Cumulative","Monthly Avg","Forecast"])

    with tabs[0]:
        st.plotly_chart(plot_price(df), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(plot_volume(df), use_container_width=True)

    with tabs[2]:
        df_ma = add_returns_and_mas(df)
        st.plotly_chart(plot_mas(df_ma), use_container_width=True)

    with tabs[3]:
        st.plotly_chart(plot_cumulative(df), use_container_width=True)

        summary = calc_yearly_cum_returns(df)
        st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]))

    with tabs[4]:
        mdf = monthly_avg_returns(df)
        st.plotly_chart(px.line(mdf, x="period", y="daily_return"), use_container_width=True)
        st.dataframe(mdf.head(12))

    with tabs[5]:
        days = st.slider("Forecast days", 7, 180, 30)
        fit = st.slider("Fit days", 10, 600, 120)

        fc = linear_forecast(df, days, fit)
        if fc is None:
            st.warning("Not enough data for forecast.")
        else:
            fig = plot_price(df)
            fig.add_trace(go.Scatter(
                x=fc["stock_date"],
                y=fc["forecast_close"],
                name="Forecast",
                line=dict(color="red", dash="dash")
            ))
            st.plotly_chart(fig, use_container_width=True)

    # Download
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name=f"{ticker}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

st.sidebar.write("---")
st.sidebar.info("Runs with SQL Server locally; uses yfinance on Streamlit Cloud.")
