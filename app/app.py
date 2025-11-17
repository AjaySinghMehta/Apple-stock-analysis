# app.py
"""
Streamlit + Plotly interactive stock analytics dashboard.
Reuses your src/db.py get_engine() for DB connection.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

# Make project root importable so `from src.db import get_engine` works
project_root = Path(__file__).resolve().parents[1]  # go one level up from /app
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------- DB import: try local DB, otherwise fall back to None ----------
USE_LOCAL_DB = False
get_engine = None

try:
    # Attempt to import local DB connector only if running locally.
    # This will fail on Streamlit Cloud if pyodbc / ODBC driver isn't available.
    # We set a flag so the rest of the app can fall back to yfinance.
    from src.db import get_engine  # local-only
    USE_LOCAL_DB = True
    st.sidebar.success("Using local SQL Server for data")
except Exception as e:
    # Running on cloud (or import failed). We'll fetch from yfinance instead.
    get_engine = None
    USE_LOCAL_DB = False
    st.sidebar.warning("Local DB not available — falling back to yfinance (cloud mode)")


st.set_page_config(page_title="Interactive Stock Dashboard", layout="wide")

# -----------------------
# Helper: fetch data
# -----------------------

@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    - If local DB available (get_engine), fetch from SQL.
    - Otherwise fetch from yfinance for cloud deployment.
    Dates should be YYYY-MM-DD strings.
    """
    # --- LOCAL DB path ---
    if USE_LOCAL_DB and get_engine is not None:
        engine = get_engine()
        query = text("""
            SELECT [stock_date], [ticker], [open], [high], [low], [close], [adj_close], [volume]
            FROM dbo.stock_daily
            WHERE ticker = :ticker AND stock_date BETWEEN :start AND :end
            ORDER BY [stock_date] ASC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ticker": ticker, "start": start_date, "end": end_date})
    else:
        # --- CLOUD path: use yfinance ---
        import yfinance as yf
        # yfinance accepts period or start/end; we'll use start/end
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        df = df.reset_index().rename(columns={
            "Date": "stock_date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        })
        df['ticker'] = ticker

    # normalize types and return
    df['stock_date'] = pd.to_datetime(df['stock_date'])
    numeric_cols = ['open','high','low','close','adj_close','volume']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# -----------------------
# Analysis helpers
# -----------------------
def add_returns_and_mas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['cumulative'] = (1 + df['daily_return']).cumprod() - 1
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA100'] = df['close'].rolling(100).mean()
    return df

def calc_yearly_cum_returns(df: pd.DataFrame):
    # returns for 1..5 years (approx 252 trading days)
    results = {}
    for y in range(1, 6):
        days = int(252 * y)
        if len(df) < 2:
            results[f"{y}Y"] = None
            continue
        sub = df.tail(days)
        val = (sub['close'].iloc[-1] / sub['close'].iloc[0]) - 1 if len(sub) > 1 else None
        results[f"{y}Y"] = float(val) if pd.notna(val) else None
    return results

def monthly_avg_returns(df: pd.DataFrame):
    df = df.copy()
    df['year'] = df['stock_date'].dt.year
    df['month'] = df['stock_date'].dt.month
    df['daily_return'] = df['close'].pct_change()
    out = (df.groupby(['year','month'])['daily_return'].mean().reset_index().assign(period=lambda d: d['year'].astype(str) + "-" + d['month'].apply(lambda m: f"{m:02d}")))
    return out[['period','daily_return']]

# simple linear trend forecast for next N days (uses last N_fit days to fit)
def linear_forecast(df: pd.DataFrame, days_forecast: int = 30, fit_days: int = 120):
    """
    Safer linear forecast:
    - Ensures fit_days <= len(df)
    - Requires at least 10 rows to fit
    - Uses business-day future dates (pd.date_range freq='B')
    - Wrapped in defensive checks; returns None on failure
    """
    try:
        df2 = df.dropna(subset=['close']).sort_values('stock_date').copy()
        n = len(df2)
        if n < 10:
            return None

        # Cap fit_days to available data, at least 10
        fit_days = int(min(max(10, fit_days), n))
        fit = df2.tail(fit_days).reset_index(drop=True)

        x = np.arange(len(fit), dtype=float)
        y = fit['close'].to_numpy(dtype=float)

        # polyfit can still fail for degenerate data; wrap it
        try:
            a, b = np.polyfit(x, y, 1)
        except Exception:
            return None

        # Forecast points (use business days)
        last_date = pd.to_datetime(fit['stock_date'].iloc[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days_forecast)
        future_x = np.arange(len(fit), len(fit) + len(future_dates))
        forecast_vals = a * future_x + b

        fc = pd.DataFrame({"stock_date": future_dates, "forecast_close": forecast_vals})
        fc['stock_date'] = pd.to_datetime(fc['stock_date'])
        return fc

    except Exception:
        return None


# -----------------------
# Plot helpers (Plotly)
# -----------------------
def plot_price(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['stock_date'], y=df['close'], name='Close', line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df['stock_date'], y=df['open'], name='Open', line=dict(width=1, dash='dash')))
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20), xaxis_title='Date', yaxis_title='Price (USD)')
    return fig

def plot_volume(df: pd.DataFrame):
    fig = px.bar(df, x='stock_date', y='volume', labels={'stock_date':'Date','volume':'Volume'})
    fig.update_layout(margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_mas(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['stock_date'], y=df['close'], name='Close', opacity=0.6))
    for ma, name in [('MA20','20-day MA'), ('MA50','50-day MA'), ('MA100','100-day MA')]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df['stock_date'], y=df[ma], name=name))
    fig.update_layout(margin=dict(l=20,r=20,t=30,b=20))
    return fig

def plot_cumulative(df):
    df = df.sort_values("stock_date")

    # Create daily returns if not already present
    if "daily_return" not in df.columns:
        df["daily_return"] = df["close"].pct_change()

    # Create cumulative returns
    df["cumulative"] = (1 + df["daily_return"]).cumprod() - 1

    fig = px.line(
        df,
        x="stock_date",
        y="cumulative",
        labels={
            "cumulative": "Cumulative Return",
            "stock_date": "Date"
        },
        title="Cumulative Return Over Selected Period"
    )
    return fig


def plot_monthly_avg(monthly_df: pd.DataFrame):
    fig = px.line(monthly_df, x='period', y='daily_return', labels={'period':'Period','daily_return':'Avg Daily Return'})
    fig.update_layout(margin=dict(l=20,r=20,t=30,b=20), xaxis_tickangle= -45)
    return fig

# -----------------------
# Streamlit UI
# -----------------------
st.title("📈 Interactive Stock Analytics (Streamlit + Plotly)")

# Sidebar controls
st.sidebar.header("Query & Options")
ticker = st.sidebar.selectbox("Ticker", ["AAPL"], index=0)  # you can add more tickers later
quick_period = st.sidebar.selectbox("Quick Range", ["5y","3y","2y","1y","6m","1m","custom"], index=0)

if quick_period != "custom":
    today = pd.Timestamp.now().normalize()
    if quick_period == "5y":
        start_date = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    elif quick_period == "3y":
        start_date = (today - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    elif quick_period == "2y":
        start_date = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    elif quick_period == "1y":
        start_date = (today - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    elif quick_period == "6m":
        start_date = (today - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    elif quick_period == "1m":
        start_date = (today - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
else:
    start_date = st.sidebar.date_input("Start date", value=(datetime.now() - timedelta(days=365)))
    end_date = st.sidebar.date_input("End date", value=datetime.now())
    # convert to strings
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

st.sidebar.write(f"Fetching: **{ticker}** from **{start_date}** → **{end_date}**")

if st.sidebar.button("Fetch & Analyze"):
    with st.spinner("Fetching data..."):
        df = fetch_stock_from_sql(ticker, start_date, end_date)
    # Top KPIs (safe version)
    if df.empty:
        st.warning("No data to display KPIs.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        start_date_val = pd.to_datetime(df['stock_date'].min())
        end_date_val   = pd.to_datetime(df['stock_date'].max())

        # Convert dates to strings for st.metric
        col1.metric("Start Date", start_date_val.strftime("%Y-%m-%d"))
        col2.metric("End Date", end_date_val.strftime("%Y-%m-%d"))

        # Prices - keep as strings with $ so metric shows correctly
        start_price = df['close'].iloc[0]
        latest_price = df['close'].iloc[-1]

        col3.metric("Start Price", f"${start_price:,.2f}")
        col4.metric("Latest Close", f"${latest_price:,.2f}")


        # Analysis options
        st.subheader(f"{ticker} Price & Analysis")
        tabs = st.tabs(["Price Trend","Volume","Moving Averages","Cumulative Return","Monthly Avg","Forecast"])

        # Price Trend
        with tabs[0]:
            fig = plot_price(df)
            st.plotly_chart(fig, use_container_width=True)

        # Volume
        with tabs[1]:
            fig = plot_volume(df)
            st.plotly_chart(fig, use_container_width=True)

        # MAs
        with tabs[2]:
            fig = plot_mas(df)
            st.plotly_chart(fig, use_container_width=True)

        # Cumulative
        with tabs[3]:
            fig = plot_cumulative(df)
            st.plotly_chart(fig, use_container_width=True)

            # show yearly cum summary
            summary = calc_yearly_cum_returns(df)
            st.write("### Yearly cumulative returns")
            st.table(pd.DataFrame.from_dict(summary, orient='index', columns=['value']).assign(value=lambda d: d['value'].map(lambda x: None if x is None else f"{x:.2%}")))

        # Monthly Avg
        with tabs[4]:
            monthly_df = monthly_avg_returns(df)
            fig = plot_monthly_avg(monthly_df)
            st.plotly_chart(fig, use_container_width=True)
            st.write("First 12 months:")
            st.dataframe(monthly_df.head(12))

        # Forecast
        with tabs[5]:
            st.write("Simple linear trend forecast (optional)")
            days = st.slider("Forecast days", min_value=7, max_value=180, value=30)
            fit_days = st.slider("Fit window (days)", min_value=10, max_value=600, value=120)
            try:
                fc = linear_forecast(df, days_forecast=days, fit_days=fit_days)
                if fc is None or fc.empty:
                    st.info("Not enough data / forecast unavailable for the chosen parameters.")
                else:
                    # combine for display
                    fig = plot_price(df)
                    fig.add_trace(
                        go.Scatter(
                            x=fc['stock_date'],
                            y=fc['forecast_close'],
                            name='Linear Forecast',
                            line=dict(color='firebrick', dash='dash')
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Forecast failed: {e}")


        # allow CSV download
        csv = df.to_csv(index=False)
        st.download_button("Download raw data (CSV)", data=csv, file_name=f"{ticker}_{start_date}_{end_date}.csv", mime="text/csv")

st.sidebar.write("---")
st.sidebar.info("This app reads from your local SQL Server and uses the DB connection in src/db.py.")
