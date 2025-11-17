"""
Quick view of top 10 records from SQL Server table (dbo.stock_daily)
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text

# --- Make project root importable ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the DB connection function
from src.db import get_engine

def main():
    print("Connecting to SQL Server...")
    engine = get_engine()
    with engine.connect() as conn:
        print("✅ Connected successfully!")

        # Fetch top 10 rows ordered by most recent date
        query = text("SELECT TOP 10 * FROM dbo.stock_daily ORDER BY stock_date DESC;")
        df = pd.read_sql(query, conn)

        print(f"\n📊 Retrieved {len(df)} rows from dbo.stock_daily\n")
        print(df)

if __name__ == "__main__":
    main()
