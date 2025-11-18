# app.py
"""
Interactive Stock Analytics
- Supports multiple tickers (multiselect)
- Uses local SQL when requested (via src.db.get_engine())
- Uses yfinance when available
- Falls back to data/*.csv files (GitHub-updated) when online fetch fails
- Works on Streamlit Cloud (set USE_LOCAL_DB=0) and locally (USE_LOCAL_DB=1)
"""

import sys
from pathlib import Path
from datetime import timedelta
import os
from typing import Optional, List
import time
from json import JSONDecodeError

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from requests import Session
from requests.exceptions import RequestException

# Make sure src package importable
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Stock Analytics", layout="wide")

# Helper: try to import local DB engine
def _try_get_engine():
    try:
        from src.db import get_engine  # type: ignore
        return get_engine
    except Exception:
        return None

USE_LOCAL_DB_DEFAULT = os.environ.get("USE_LOCAL_DB", "0") in ("1", "true", "True")

# ---------- DEFAULT TICKERS (FAANG + big tech)
DEFAULT_TICKERS = [
    "AAPL",  # Apple
    "META",  # Meta (Facebook)
    "AMZN",  # Amazon
    "GOOGL", # Alphabet
    "NFLX",  # Netflix
    "MSFT",  # Microsoft
    "TSLA"   # Tesla
]

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

# ---------- Fetch / normalize helpers (SQL / yfinance / CSV fallback) ----------
@st.cache_data(ttl=300)
def fetch_from_sql(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    get_engine = _try_get_engine()
    if not get_engine:
        return pd.DataFrame()
    try:
        from sqlalchemy import text
        engine = get_engine()
        query = text("""
            SELECT stock_date, ticker, open, high, low, close, adj_close, volume
            FROM dbo.stock_daily
            WHERE ticker = :ticker AND stock_date BETWEEN :start AND :end
            ORDER BY stock_date ASC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "ticker": ticker,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d")
            })
        return _normalize_df(df, ticker)
    except Exception as e:
        st.write("Debug (SQL fetch):", repr(e))
        return pd.DataFrame()

def load_csv_fallback(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load from data/<TICKER>.csv if exists. Trim to date range.
    """
    path = DATA_FOLDER / f"{ticker.upper()}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["stock_date"])
        df = _normalize_df(df, ticker)
        # filter by range
        mask = (df["stock_date"] >= start) & (df["stock_date"] <= end)
        return df.loc[mask].reset_index(drop=True)
    except Exception as e:
        st.write("Debug (CSV load):", repr(e))
        return pd.DataFrame()

def fetch_from_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Robust yfinance fetch with retries. Returns normalized df or empty df.
    """
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    sess = Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
    })

    yf_end = end + pd.Timedelta(days=1)
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
            if raw is None or raw.empty:
                raise ValueError("yfinance returned empty dataframe")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            return _normalize_df(raw, ticker)
        except (JSONDecodeError, ValueError, RequestException, Exception) as e:
            msg = f"yfinance attempt {attempt} failed: {repr(e)}"
            # small message to UI/logs
            print(msg)
            st.write(msg)
            if attempt < attempts:
                time.sleep(backoff)
                backoff *= 2
                continue

    # fallback to history()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=yf_end.strftime("%Y-%m-%d"), interval="1d")
        if hist is not None and not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] for c in hist.columns]
            return _normalize_df(hist, ticker)
    except Exception as e:
        st.write("yfinance.history fallback failed:", repr(e))
        print("yfinance.history fallback failed:", repr(e))

    return pd.DataFrame()

def _normalize_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize columns and ensure canonical column names.
    """
    df = df.copy()
    # index may be datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # unify column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # rename common variants to stock_date
    if "stock_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "stock_date"})

    # try to detect a datetime column if missing
    if "stock_date" not in df.columns:
        for col in df.columns:
            try:
                sample = pd.to_datetime(df[col].iloc[:5], errors="coerce")
                if sample.notna().any():
                    df = df.rename(columns={col: "stock_date"})
                    break
            except Exception:
                continue

    if "stock_date" not in df.columns:
        df["stock_date"] = pd.NaT

    # canonicalize adj close
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

    # ensure required columns exist
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df["stock_date"] = pd.to_datetime(df["stock_date"], errors="coerce").dt.normalize()
    if "ticker" not in df.columns:
        df["ticker"] = ticker.upper()

    df = df.sort_values("stock_date", na_position="last").reset_index(drop=True)
    return df

def fetch_stock_for_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp, use_local: bool) -> pd.DataFrame:
    """
    High-level fetch for one ticker:
    - try local SQL (if enabled)
    - then yfinance
    - then local CSV fallback (data/<TICKER>.csv)
    """
    # 1) Local SQL (if user asked)
    if use_local:
        df_sql = fetch_from_sql(ticker, start, end)
        if not df_sql.empty:
            return df_sql

    # 2) Try yfinance
    df_yf = fetch_from_yfinance(ticker, start, end)
    if not df_yf.empty:
        return df_yf

    # 3) CSV fallback
    df_csv = load_csv_fallback(ticker, start, end)
    if not df_csv.empty:
        return df_csv

    # nothing worked
    return pd.DataFrame()

# ---------- analysis helpers ----------
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

# ---------- plotting helpers ----------
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

# ---------- UI ----------
st.title("📊 Interactive Stock Analytics (SQL + Streamlit + Plotly)")

st.sidebar.header("Query Options")
tickers = st.sidebar.multiselect("Tickers", options=DEFAULT_TICKERS, default=["AAPL"], help="Select one or more tickers.")
# let user add custom tickers in a text input (comma-separated)
custom = st.sidebar.text_input("Add tickers (comma separated)", value="")
if custom.strip():
    extras = [t.strip().upper() for t in custom.split(",") if t.strip()]
    for t in extras:
        if t not in tickers:
            tickers.append(t)

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

st.sidebar.write(f"Date range: {start.date()} → {end.date()}")

fetch_btn = st.sidebar.button("Fetch & Analyze")

# quick debug fetch button
if st.sidebar.button("Quick debug fetch (AAPL)"):
    df_dbg = fetch_stock_for_ticker("AAPL", pd.to_datetime("2020-01-01"), pd.Timestamp.today(), use_local=False)
    st.write("DEBUG columns:", df_dbg.columns.tolist())
    st.write(df_dbg.head(3))

# When fetch is clicked, retrieve data for each ticker (one by one) and store in session state
if fetch_btn:
    if not tickers:
        st.error("Select at least one ticker.")
        st.stop()

    all_data = {}
    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers, start=1):
        st.info(f"Fetching {t} ...")
        df_t = fetch_stock_for_ticker(t, start, end, use_local)
        if df_t.empty:
            st.warning(f"No data for {t} (tried SQL, yfinance, CSV).")
        else:
            df_t = add_indicators(df_t)
            all_data[t] = df_t
        progress.progress(int(i / total * 100))

    if not all_data:
        st.error("No data found for the selected tickers / date ranges.")
        st.session_state["multi_data"] = {}
    else:
        st.session_state["multi_data"] = all_data
        st.success("Fetched data for: " + ", ".join(all_data.keys()))

# load stored data
multi_data = st.session_state.get("multi_data", {})

if not multi_data:
    st.info("No data loaded yet. Use the sidebar to select tickers and click Fetch & Analyze.")
    st.caption("Tip: Add extra tickers (comma-separated) in the sidebar.")
    st.stop()

# If we have multiple tickers, show a selectbox to choose which to view
selected_ticker = st.selectbox("Select ticker to view", options=list(multi_data.keys()))
df = multi_data[selected_ticker]

# KPIs
st.subheader(f"📌 Key Metrics — {selected_ticker}")
c1, c2, c3, c4 = st.columns(4)
try:
    c1.metric("Start Date", df["stock_date"].min().strftime("%Y-%m-%d"))
    c2.metric("End Date", df["stock_date"].max().strftime("%Y-%m-%d"))
except Exception:
    c1.metric("Start Date", "N/A")
    c2.metric("End Date", "N/A")
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

tabs = st.tabs(["Price Trend", "Volume", "Moving Avg", "Cumulative", "Monthly Avg", "Forecast", "Download"])

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

with tabs[6]:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"{selected_ticker}_stock.csv", mime="text/csv")

# End of app.py
