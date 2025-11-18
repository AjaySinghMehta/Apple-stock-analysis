# src/db.py
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine

# load .env if present
load_dotenv()

def get_engine(echo: bool = False):
    """
    Create a SQLAlchemy engine for SQL Server using pyodbc.
    Reads config from environment variables.
    """
    server = os.getenv("DB_SERVER", "localhost")
    database = os.getenv("DB_NAME", "apple_stock")
    username = os.getenv("DB_USER", "sa")
    password = os.getenv("DB_PASSWORD", "StrongPassword123#")
    driver = os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server")

    if not server or not database:
        raise ValueError("Missing DB_SERVER or DB_NAME environment variable.")

    # Quote username/password to be safe with special chars
    user_enc = quote_plus(username)
    pwd_enc = quote_plus(password)
    driver_enc = quote_plus(driver)

    # Example: mssql+pyodbc://sa:pwd@localhost/apple_stock?driver=ODBC+Driver+17+for+SQL+Server
    connection_url = f"mssql+pyodbc://{user_enc}:{pwd_enc}@{server}/{database}?driver={driver_enc}&Encrypt=no"

    try:
        engine = create_engine(
            connection_url,
            fast_executemany=True,
            echo=echo,
            pool_pre_ping=True  # helps avoid "MySQL has gone away" style issues
        )
        return engine
    except Exception as e:
        # raise a clear error for debugging (won't leak secrets)
        raise RuntimeError(f"Failed to create SQL engine: {e}")
