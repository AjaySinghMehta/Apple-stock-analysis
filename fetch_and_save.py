# fetch_and_save.py
"""
Download stock CSVs for a list of tickers and save as data/<TICKER>.csv
This script is intended to be run by GitHub Actions (or locally).
"""
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

# tickers to fetch (same set as default plus you can add more)
TICKERS = ["AAPL", "META", "AMZN", "GOOGL", "NFLX", "MSFT", "TSLA"]

# Fetch range: past 10 years by default (adjust as needed)
END = datetime.utcnow().date()
START = END - timedelta(days=3650)  # ~10 years

def fetch_and_save(ticker: str):
    print(f"Fetching {ticker} ...")
    df = yf.download(ticker, start=START.strftime("%Y-%m-%d"), end=(END + timedelta(days=1)).strftime("%Y-%m-%d"), threads=False)
    if df is None or df.empty:
        print(f"Warning: no data for {ticker}")
        return
    # collapse multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "stock_date", "Adj Close": "adj_close"})
    # ensure canonical column order
    cols = ["stock_date", "Open", "High", "Low", "Close", "adj_close", "Volume"]
    # map to lowercase names to match app normalization (we save with stock_date column)
    df.columns = [c if c != "Adj Close" else "adj_close" for c in df.columns]
    # Save with canonical column name 'stock_date'
    df.to_csv(DATA_FOLDER / f"{ticker}.csv", index=False)
    print(f"Saved {ticker} to {DATA_FOLDER / f'{ticker}.csv'}")

if __name__ == "__main__":
    for tk in TICKERS:
        try:
            fetch_and_save(tk)
        except Exception as e:
            print("Error fetching", tk, e)
