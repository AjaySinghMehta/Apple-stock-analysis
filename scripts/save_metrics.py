# scripts/save_metrics.py
"""
Save computed Apple stock metrics (cumulative & monthly returns) to SQL Server.
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from datetime import datetime

# Add project root for imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.db import get_engine
from scripts.analyze_stock import load_data, calculate_cumulative_returns

def save_metrics_to_sql(summary: dict, monthly_avg: pd.DataFrame, ticker: str = "AAPL"):
    """
    summary: dict like {"1Y_cum_return": 0.20, ...}
    monthly_avg: DataFrame with columns ['year', 'month', 'daily_return']
    """
    print("🧾 Saving metrics to SQL Server...")
    engine = get_engine()

    # Format yearly summary -> list of dicts
    yearly_rows = [
        {
            "ticker": ticker,
            "period_type": "yearly",
            "period_label": period,                # e.g. "1Y_cum_return"
            "return_value": float(ret),            # convert numpy types to python float
            "created_at": datetime.now()
        }
        for period, ret in summary.items()
    ]

    # Format monthly averages -> list of dicts
    monthly_rows = []
    # Ensure monthly_avg has expected columns
    if not monthly_avg.empty:
        # In case 'monthly_avg' contains 'year','month','daily_return'
        for _, row in monthly_avg.iterrows():
            y = int(row["year"])
            m = int(row["month"])
            label = f"{y}-{m:02d}"  # e.g. "2024-07"
            val = float(row["daily_return"]) if pd.notna(row["daily_return"]) else None
            monthly_rows.append({
                "ticker": ticker,
                "period_type": "monthly",
                "period_label": label,
                "return_value": val,
                "created_at": datetime.now()
            })

    combined_df = pd.DataFrame(yearly_rows + monthly_rows)

    # Defensive: if combined_df is empty, stop
    if combined_df.empty:
        print("⚠️ Nothing to save (combined_df is empty). Exiting.")
        return

    # Make sure columns are correct order and types
    combined_df = combined_df[["ticker", "period_type", "period_label", "return_value", "created_at"]]
    combined_df["return_value"] = pd.to_numeric(combined_df["return_value"], errors="coerce")
    combined_df["created_at"] = pd.to_datetime(combined_df["created_at"])

    print(f"Saving {len(combined_df)} rows...")
    print(combined_df.head())

    # Write to SQL (append)
    with engine.begin() as conn:
        combined_df.to_sql(
            name="stock_metrics",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False
        )

    print("✅ Metrics saved successfully!")


def main():
    df = load_data()
    summary, monthly_avg = calculate_cumulative_returns(df)

    # print summary and a sample of monthly_avg for verification
    print("\n===== YEARLY CUMULATIVE RETURN SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v:.2%}")

    print("\n===== MONTHLY AVERAGE RETURNS (first 10) =====")
    print(monthly_avg.head(10))

    save_metrics_to_sql(summary, monthly_avg, ticker="AAPL")


if __name__ == "__main__":
    main()
