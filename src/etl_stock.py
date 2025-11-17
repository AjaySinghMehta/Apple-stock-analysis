"""
ETL pipeline for Apple stock data.
Steps:
1 - Extract data from Yahoo Finance using yfinance
2 - Transform/clean using pandas.
3 - Load to SQL Server (dbo.stock_daily) via SQLAlchemy
"""

import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from sqlalchemy import text

#--ensuring src and root are importable ---

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from src.db import get_engine # importing our DB connection engine

#Extracting - Get Apple stock data

def extract_stock_data(ticker="AAPL", period="5y"):
    print(f"Extracting {ticker} data for {period}")
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    data.reset_index(inplace=True)

    # ✅ Flatten multi-index columns (fix for tuple column names)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    return data



##Transformig the data - cleaning and formating data

def transform_data(df):
    print("Transforming data...")
    df = df.rename(
        columns = {
            "Date" : "stock_date",
            "Open" : "open",
            "High" : "high",
            "Low" : "low",
            "Close" : "close",
            "Adj Close" : "adj_close",
            "Volume" : "volume"
        }
    )
    
    # adding ticker column to match SQL Schema
    df["ticker"] = "AAPL"
    # Ensuring correct data types
    df["stock_date"] = pd.to_datetime(df["stock_date"])
    numeric_cols = ["open","high","low","close","adj_close","volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors = "coerce")

    #Reordering columns to match SQL table
    df = df[["stock_date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    print(f"cleaned {len(df)} rows.")
    return df


#Loading the data to sql server - Saving the loaded data to csv and sql server 

def load_to_sql(df, table_name = "stock_daily"):
    print(f"loading data into SQL Server table: {table_name}")
    engine = get_engine()
    
    # Delete only overlapping dates before inserting new ones
    # with engine.begin() as conn:
    #     min_date = df["stock_date"].min()
    #     max_date = df["stock_date"].max()
    #     conn.execute(
    #         text(f"""
    #             DELETE FROM dbo.{table_name}
    #             WHERE stock_date BETWEEN :min_d AND :max_d
    #         """),
    #         {"min_d": min_date, "max_d": max_date}
    #     )

    with engine.begin() as conn:
        # we can also clear old data for fresh load
        #conn.execute(text(f"TRUNCATE TABLE dbo.{table_name}"))
        
        df.to_sql(
            name = table_name,
            con = conn,
            if_exists = "append" , # appending to add data, or we can use replace to recreate
            index = False,
            schema = "dbo"
        )
    print("Data successfully loaded into SQL Server..")
    
    
    # main function
    
def main():
    raw_data = extract_stock_data("AAPL","5y")
    clean_data = transform_data(raw_data)
    
    # saving to local csv as backup
    output_dir = project_root/"data"/"processed"
    output_dir.mkdir(parents = True, exist_ok = True)
    csv_path = output_dir/"apple_stock_clean.csv"
    clean_data.to_csv(csv_path, index = False)
    print(f"Saved cleaned CSV : {csv_path}")
    
    load_to_sql(clean_data)
    print("ETL completed successfully!")
        
if __name__ == "__main__":
    main()