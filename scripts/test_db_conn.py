# # scripts/test_db_conn.py

# """
# Quick test script to verify DB connection and table access.
# """
# import sys
# from pathlib import Path

# # Add project root (one folder up from "scripts") to the Python path
# sys.path.append(str(Path(__file__).resolve().parents[1]))


# from src.db import get_engine
# import pandas as pd

# def main():
#     print("Connecting to SQL Server...")

#     # Get the SQLAlchemy engine from db.py
#     engine = get_engine(echo=True)

#     # Open connection safely using context manager
#     with engine.connect() as conn:
#         print("✅ Connection successful!")

#         # Verify database details
#         result = conn.execute("SELECT DB_NAME() AS current_db, @@VERSION AS sql_version;")
#         for row in result:
#             print(f"Connected to database: {row.current_db}")
#             print(f"SQL Server version: {row.sql_version.split()[0:3]}")

#         # Try reading data from your table (even if empty)
#         try:
#             df = pd.read_sql("SELECT TOP 5 * FROM dbo.stock_daily ORDER BY stock_date DESC", conn)
#             print(f"\n✅ Table read successful — {len(df)} rows returned")
#             print(df.head())
#         except Exception as e:
#             print("\n⚠️ Could not read data (probably empty table):", e)

# if __name__ == "__main__":
#     main()

# scripts/test_db_conn.py
"""
Test DB connection in a robust way:
- Ensures project root is on sys.path so 'src' imports work
- Uses sqlalchemy.text() for SQLAlchemy 2.x compliant execution
"""

import sys
from pathlib import Path

# --- make project root importable (works even if you run the script directly) ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now normal imports
from src.db import get_engine
from sqlalchemy import text
import pandas as pd

def main():
    print("Connecting to SQL Server...")
    engine = get_engine(echo=True)   # echo=True prints SQL statements for debug

    try:
        with engine.connect() as conn:
            print("✅ Connection successful!")

            # SQLAlchemy 2.x requires text() for literal SQL
            q = text("SELECT DB_NAME() AS current_db, @@VERSION AS sql_version;")
            result = conn.execute(q)
            for row in result:
                print(f"📊 Current DB: {row.current_db}")
                print(f"🧠 SQL Server Version:\n{row.sql_version}")

            # Optional: try reading top 5 rows from your table (safe even if empty)
            try:
                df = pd.read_sql(text("SELECT TOP 5 * FROM dbo.stock_daily ORDER BY stock_date DESC"), conn)
                print(f"\n✅ Table read successful — {len(df)} rows returned")
                print(df.head())
            except Exception as e:
                print("\n⚠️ Could not read data (probably empty or table missing):", e)

    except Exception as e:
        print("❌ Error connecting or executing:", repr(e))

if __name__ == "__main__":
    main()
