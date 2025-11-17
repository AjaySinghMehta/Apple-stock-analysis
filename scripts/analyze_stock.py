"""Analyze apple stock data from SQL Server
1 : Load from SQL (dbo.stock_daily)
2 : Perform basic EDA (Exploratory Data Analysis)
3 : Visualize trends and patterns
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text
# import datetime as dt
# Adding project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from src.db import get_engine

def load_data():
    """
    Fetch Apple Stock data from SQL Server
    """
    print("Fetching data from SQL Server...")
    engine = get_engine()
    query = text("SELECT * FROM dbo.stock_daily ORDER BY stock_date ASC")
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Loaded {len(df)} rows.")
    print(df.head())
    return df

def basic_analysis(df):
    """ Basic checks and summary statistics"""
    print("\n summary statistics : ")
    print(df.describe())
    
    print("\n Date range : ", df['stock_date'].min(), "->" , df['stock_date'].max())
    print("total trading days :", len(df))
    
def visualize_trends(df):
    """Plot key stock trends"""
    print("\n Plotting stock price trends...")
    
    plt.figure(figsize=(10,5))
    plt.plot(df['stock_date'],df['close'], label = 'Close Price', linewidth = 1.5)
    plt.plot(df['stock_date'], df['open'], label = 'Open Price', linestyle = '--', linewidth = 1)
    plt.title("Apple Stock Price (5 Years)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def visualize_volume(df):
    """Plotting daily trading volume"""
    print("\n Plotting trading volume trend...")
    
    plt.figure(figsize=(10,4))
    plt.bar(df['stock_date'], df['volume'], color = 'gray', alpha = 0.6)
    plt.title("Apple Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()


def visualize_moving_averages(df):
    """Plot closing price with moving averages"""
    print("\n📈 Plotting moving averages...")

    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA100'] = df['close'].rolling(window=100).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df['stock_date'], df['close'], label='Close Price', linewidth=1)
    plt.plot(df['stock_date'], df['MA20'], label='20-Day MA', linewidth=1.5)
    plt.plot(df['stock_date'], df['MA100'], label='100-Day MA', linewidth=2)
    plt.title("Apple Stock - Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_cumulative_returns(df):
    """Calculate cumulative returns for 1Y, 2Y, 3Y, 4Y, 5Y and monthly averages."""
    
    df['stock_date'] = pd.to_datetime(df['stock_date'], errors='coerce')
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    # Helper to get trading-day windows
    def cumulative_for_years(df, years):
        days = int(years * 252)  # ~252 trading days per year
        if len(df) > days:
            subset = df.tail(days)
        else:
            subset = df
        return (subset['close'].iloc[-1] / subset['close'].iloc[0]) - 1

    summary = {
        "1Y_cum_return": cumulative_for_years(df, 1),
        "2Y_cum_return": cumulative_for_years(df, 2),
        "3Y_cum_return": cumulative_for_years(df, 3),
        "4Y_cum_return": cumulative_for_years(df, 4),
        "5Y_cum_return": cumulative_for_years(df, 5)
    }

    # Monthly average return
    df['year'] = df['stock_date'].dt.year
    df['month'] = df['stock_date'].dt.month
    monthly_avg = (
        df.groupby(['year', 'month'])['daily_return'].mean().reset_index().sort_values(['year', 'month'])
    )

    print("\n📈 Yearly cumulative returns:")
    for k, v in summary.items():
        print(f"  {k}: {v:.2%}")

    print("\n📊 Monthly average returns (first 10):")
    print(monthly_avg.head(10))

    return summary, monthly_avg


def main():
    df = load_data()
    basic_analysis(df)
    visualize_trends(df)
    visualize_volume(df)
    visualize_moving_averages(df)
    summary, monthly_avg = calculate_cumulative_returns(df)
    # Print summary
    print("\n===== YEARLY CUMULATIVE RETURN SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v:.2%}")

    # Print a few rows of monthly averages
    print("\n===== MONTHLY AVERAGE RETURNS (first 10) =====")
    print(monthly_avg.head(10))

if __name__ == "__main__":
    main()    