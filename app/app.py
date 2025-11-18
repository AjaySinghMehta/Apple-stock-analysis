# app.py
"""
Complete Streamlit app for Apple stock analytics.
- Works with local SQL Server via src.db.get_engine() (set USE_LOCAL_DB=1 in .env)
- Works on Streamlit Cloud using yfinance fallback (set USE_LOCAL_DB=0 on cloud)
- Robust yfinance fetching with retries and history() fallback
- Normalizes yfinance MultiIndex columns and guards missing columns
- Uses width="stretch" (Streamlit deprecation-safe)
"""

import sys
from pathlib import Path
from datetime import timedelta
import os
from typing import Optional
import time
from json import JSONDecodeError

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from requests import Session
from requests.exceptions import RequestException

# Ensure project root in path so `from src.db import get_engine` can work
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Stock Analytics", layout="wide")

# ---- helper to optionally load local DB engine ----
def _try_get_engine():
    try:
        from src.db import get_engine  # type: ignore
        return get_engine
    except Exception:
        return None

USE_LOCAL_DB_DEFAULT = os.environ.get("USE_LOCAL_DB", "0") in ("1", "true", "True")

# ---- robust fetch_stock (retries + session + fallback) ----
@st.cache_data(ttl=300)
def fetch_stock(ticker: str, start, end, use_local: bool = False) -> pd.DataFrame:
    """
    Returns canonical DataFrame with columns:
    ['stock_date','ticker','open','high','low','close','adj_close','volume']
    Tries local SQL (if requested & available) otherwise uses yfinance with retries and fallback.
    """
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        # If index is datetime, reset it to a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # Lowercase and normalize column names
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # Map date column variants
        if "stock_date" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "stock_date"})

        # If still no stock_date, try to detect a datetime-like column
        if "stock_date" not in df.columns:
            for col in df.columns:
                try:
                    sample = pd.to_datetime(df[col].iloc[:5], errors="coerce")
                    if sample.notna().any():
                        df = df.rename(columns={col: "stock_date"})
                        break
                except Exception:
                    continue

        # Ensure stock_date exists
        if "stock_date" not in df.columns:
            df["stock_date"] = pd.NaT

        # Normalize adj_close variants and other column names
        col_map = {}
        for src in list(df.columns):
            if src in ("open", "high", "low", "close", "volume"):
                col_map[src] = src
            if "adj" in src and "close" in src:
                col_map[src] = "adj_close"
            if src == "adjclose":
                col_map[src] = "adj_close"
        if col_map:
            df = df.rename(columns=col_map)

        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        # Normalize date column
        df["stock_date"] = pd.to_datetime(df["stock_date"], errors="coerce").dt.normalize()

        # Ensure ticker exists
        if "ticker" not in df.columns:
            df["ticker"] = ticker

        # Sort and reset index
        df = df.sort_values("stock_date", na_position="last").reset_index(drop=True)
        return df

    # --- Try local SQL if requested ---
    if use_local:
        get_engine = _try_get_engine()
        if get_engine:
            try:
                from sqlalchemy import text
                engine = get_engine()
                query = text(
                    """
                    SELECT stock_date, ticker, open, high, low, close, adj_close, volume
                    FROM dbo.stock_daily
                    WHERE ticker = :ticker AND stock_date BETWEEN :start AND :end
                    ORDER BY stock_date ASC
                    """
                )
                with engine.connect() as conn:
                    df_sql = pd.read_sql(query, conn, params={
                        "ticker": ticker,
                        "start": start.strftime("%Y-%m-%d"),
                        "end": end.strftime("%Y-%m-%d")
                    })
                return _normalize_df(df_sql)
            except Exception as e:
                st.info("Local SQL Server unavailable or returned unexpected columns. Falling back to yfinance.")
                # short debug message (safe)
                st.write("Debug (local SQL):", repr(e))

    # --- yfinance fallback with retries ---
    try:
        import yfinance as yf
    except Exception:
        st.error("yfinance not available. Install yfinance or enable local DB.")
        return pd.DataFrame(columns=["stock_date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    # prepare a requests Session (helps in some environments)
    sess = Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    })

    yf_end = end + pd.Timedelta(days=1)  # inclusive end
    attempts = 3
    backoff = 1.0

    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=yf_end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                threads=False,
                session=sess
            )
            # treat empty as failure for retry
            if raw is None or raw.empty:
                raise ValueError("yfinance returned empty dataframe")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            return _normalize_df(raw)
        except (JSONDecodeError, ValueError, RequestException, Exception) as e:
            msg = f"yfinance attempt {attempt} failed: {repr(e)}"
            st.write(msg)
            print(msg)
            if attempt < attempts:
                time.sleep(backoff)
                backoff *= 2
                continue

    # Final fallback: try yf.Ticker.history()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=yf_end.strftime("%Y-%m-%d"), interval="1d")
        if hist is not None and not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] for c in hist.columns]
            return _normalize_df(hist)
    except Exception as e:
        st.write("yfinance.history fallback failed:", repr(e))
        print("yfinance.history fallback failed:", repr(e))

    st.error("Failed to fetch data from yfinance. Please try again later or use local DB.")
    return pd.DataFrame(columns=["stock_date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])


# ---- analysis helpers ----
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("stock_date")
    if "close" not in df.columns or df["close"].dropna().empty:
        for col in ["daily_return", "cumulative", "MA20", "MA50", "MA100"]:
            df[col] = np.nan
        return df
    df["daily_return"] = df["close"].pct_change()
    df["cumulative"] = (1 + df["daily_return"]).cumprod() - 1
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA100"] = df["close"].rolling(100).mean()
    return df


def yearly_returns(df: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for y in range(1, 6):
        days = int(252 * y)
        if len(df) > days and "close" in df.columns and df["close"].dropna().shape[0] > days:
            sub = df.tail(days)
            r = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
            result[f"{y}Y"] = f"{r:.2%}"
        else:
            result[f"{y}Y"] = "N/A"
    return pd.DataFrame.from_dict(result, orient="index", columns=["Return"])


def monthly_avg(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "stock_date" not in df2.columns:
        return pd.DataFrame(columns=["period", "daily_return"])
    df2["year"] = df2["stock_date"].dt.year
    df2["month"] = df2["stock_date"].dt.month
    if "daily_return" not in df2.columns:
        df2["daily_return"] = df2.get("close", pd.Series(dtype=float)).pct_change()
    avg = df2.groupby(["year", "month"])["daily_return"].mean().reset_index()
    avg["period"] = avg["year"].astype(str) + "-" + avg["month"].astype(str).str.zfill(2)
    return avg[["period", "daily_return"]].sort_values("period")


def linear_forecast(df: pd.DataFrame, n_days: int = 30, fit_days: int = 120) -> Optional[pd.DataFrame]:
    df = df.dropna(subset=["close"]).sort_values("stock_date")
    if len(df) < 30:
        return None
    fit_days = min(fit_days, len(df))
    sub = df.tail(fit_days).reset_index(drop=True)
    x = np.arange(len(sub))
    y = sub["close"].values
    try:
        a, b = np.polyfit(x, y, 1)
    except Exception:
        return None
    start_date = sub["stock_date"].iloc[-1] + pd.Timedelta(days=1)
    future_dates = pd.bdate_range(start=start_date, periods=n_days)
    fx = np.arange(len(sub), len(sub) + n_days)
    fy = a * fx + b
    return pd.DataFrame({"stock_date": future_dates.normalize(), "forecast_close": fy})


# ---- plotting helpers ----
def p_price(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "stock_date" in df.columns and "close" in df.columns:
        fig.add_trace(go.Scatter(x=df["stock_date"], y=df["close"], name="Close", mode="lines"))
    if "stock_date" in df.columns and "open" in df.columns:
        fig.add_trace(go.Scatter(x=df["stock_date"], y=df["open"], name="Open", mode="lines", line=dict(dash="dash")))
    fig.update_layout(legend_title_text="Price")
    return fig


def p_volume(df: pd.DataFrame):
    if "stock_date" not in df.columns or "volume" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title_text="No volume data available for this selection.")
        return fig
    return px.bar(df, x="stock_date", y="volume", labels={"volume": "Volume", "stock_date": "Date"})


def p_ma(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "stock_date" in df.columns and "close" in df.columns:
        fig.add_trace(go.Scatter(x=df["stock_date"], y=df["close"], name="Close"))
    for col in ["MA20", "MA50", "MA100"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["stock_date"], y=df[col], name=col))
    fig.update_layout(legend_title_text="Moving Averages")
    return fig


def p_cum(df: pd.DataFrame):
    if "stock_date" not in df.columns or "cumulative" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title_text="No cumulative return data available.")
        return fig
    return px.line(df, x="stock_date", y="cumulative", labels={"cumulative": "Cumulative Return", "stock_date": "Date"})


# ---- UI ----
st.title("📊 Interactive Stock Analytics (SQL + Streamlit + Plotly)")
st.sidebar.header("Query Options")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
range_opt = st.sidebar.selectbox("Range", ["5y", "3y", "1y", "6m", "1m", "custom"], index=0)
use_local = st.sidebar.checkbox("Use Local SQL Server", value=USE_LOCAL_DB_DEFAULT)

today = pd.Timestamp.today().normalize()

if range_opt == "5y":
    start = today - pd.DateOffset(years=5)
elif range_opt == "3y":
    start = today - pd.DateOffset(years=3)
elif range_opt == "1y":
    start = today - pd.DateOffset(years=1)
elif range_opt == "6m":
    start = today - pd.DateOffset(months=6)
elif range_opt == "1m":
    start = today - pd.DateOffset(months=1)
else:
    s = st.sidebar.date_input("Start Date", today - pd.DateOffset(years=1))
    start = pd.to_datetime(s).normalize()

end = today

st.sidebar.write(f"Fetching **{ticker}** from {start.date()} → {end.date()}")

fetch_btn = st.sidebar.button("Fetch & Analyze")

# Debug button (temporary) - click to run a quick AAPL fetch and print debug info
if st.sidebar.button("Quick debug fetch (AAPL)"):
    debug_df = fetch_stock("AAPL", "2020-01-01", pd.Timestamp.today(), use_local=False)
    st.write("DEBUG - Columns:", debug_df.columns.tolist())
    st.write(debug_df.head(3))

# session state for last df
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
    st.session_state["last_ticker"] = None
    st.session_state["last_range"] = None

if fetch_btn:
    with st.spinner(f"Fetching {ticker} data ..."):
        df = fetch_stock(ticker, start, end, use_local=use_local)
    if df.empty:
        st.error("No data found for the selected ticker / date range.")
        st.session_state["last_df"] = None
    else:
        df = add_indicators(df)
        st.session_state["last_df"] = df
        st.session_state["last_ticker"] = ticker
        st.session_state["last_range"] = (start, end)

df = st.session_state.get("last_df", None)

if df is not None:
    st.subheader(f"📌 Key Metrics — {st.session_state.get('last_ticker','')}")
    c1, c2, c3, c4 = st.columns(4)

    # Safe start/end
    try:
        c1.metric("Start Date", df["stock_date"].min().strftime("%Y-%m-%d"))
        c2.metric("End Date", df["stock_date"].max().strftime("%Y-%m-%d"))
    except Exception:
        c1.metric("Start Date", "N/A")
        c2.metric("End Date", "N/A")

    # Safe prices
    if "close" in df.columns and not df["close"].dropna().empty:
        try:
            c3.metric("Start Price", f"${df['close'].iloc[0]:,.2f}")
            c4.metric("Latest Close", f"${df['close'].iloc[-1]:,.2f}")
        except Exception:
            c3.metric("Start Price", "N/A")
            c4.metric("Latest Close", "N/A")
    else:
        c3.metric("Start Price", "N/A")
        c4.metric("Latest Close", "N/A")

    tabs = st.tabs(["Price Trend", "Volume", "Moving Avg", "Cumulative", "Monthly Avg", "Forecast"])

    with tabs[0]:
        st.plotly_chart(p_price(df), width="stretch")

    with tabs[1]:
        st.plotly_chart(p_volume(df), width="stretch")

    with tabs[2]:
        st.plotly_chart(p_ma(df), width="stretch")

    with tabs[3]:
        st.plotly_chart(p_cum(df), width="stretch")
        st.subheader("Yearly Returns")
        st.table(yearly_returns(df))

    with tabs[4]:
        m = monthly_avg(df)
        if not m.empty:
            st.plotly_chart(px.line(m, x="period", y="daily_return", labels={"daily_return": "Avg Daily Return", "period": "Period"}), width="stretch")
            st.dataframe(m.tail(36))
        else:
            st.info("No monthly average data available for this dataset.")

    with tabs[5]:
        days = st.slider("Forecast Days", 7, 180, 30)
        fit = st.slider("Fit Window (Days)", 30, 400, 120)
        fc = linear_forecast(df, days, fit)
        if fc is None:
            st.warning("Not enough data for forecast or forecast failed.")
        else:
            fig = p_price(df)
            fig.add_trace(go.Scatter(x=fc["stock_date"], y=fc["forecast_close"], name="Forecast", line=dict(color="red")))
            st.plotly_chart(fig, width="stretch")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"{ticker}_stock.csv", mime="text/csv")
else:
    st.info("No data loaded yet. Choose options in the sidebar and click **Fetch & Analyze**.")
    st.caption("If you want to load from your local SQL Server, check 'Use Local SQL Server' and ensure src/db.py exists with get_engine().")
