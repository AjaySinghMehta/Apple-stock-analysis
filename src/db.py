# """
# Database connection helper for the project

# - Read credentials from .env
# - Builds a SQLAlchemy engine for SQL Server via pyodbc
# - Exposes get_engine() which other modules import and reuse    
# """
# import os
# from urllib.parse import quote_plus

# from dotenv import load_dotenv
# from sqlalchemy import create_engine

# # load environment variables from .env file into the process environment
# load_dotenv()

# #Read configuration values from environment(with sensible defaults)
# DB_HOST = os.getenv("DB_HOST","localhost")
# DB_PORT = os.getenv("DB_PORT","1433")
# DB_NAME = os.getenv("DB_NAME", "sales_analytics")
# DB_USER = os.getenv("DB_USER","sa")
# DB_PASSWORD = os.getenv("DB_PASSWORD","")
# ODBC_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

# def _build_odbc_connection_string():
#     """
#     Build a raw ODBC connection string and URL-encode it for SQLAlchemy.
#     Example output (before encoding):
#         Driver = {ODBC Driver 17 for SQL Server};
#         SERVER = localhost, 1433;
#         DATABASE = sales_analytics;
#         UID = sa;
#         PWD = xxx;
#         Encrypt = no
        
#     """
#     # Use Server = host, port for default instance; for named instance you may use SERVER = Localhost\SQLEXPRESS
#     odbc_str = (
#         f"DRIVER={{{ODBC_DRIVER}}};"
#         f"SERVER={DB_HOST},{DB_PORT};"
#         f"DATABASE={DB_NAME};"
#         f"UID={DB_USER};"
#         f"PWD={DB_PASSWORD};"
#         "Encrypt=no;"
#         "TrustServerCertificate=yes;"
#     )
#     print("🔍 Raw ODBC string before encoding:\n", odbc_str)  # <-- for debugging
#     return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
#     # return quote_plus(odbc_str)


# def get_engine(echo : bool = False):
#     """
#     Return a SQLAlchemy engine connected to the configured SQL Server.
#     - echo = True prints SQL statements (handy for debugging).
#     - fast_executemany = True speeds up bulk inserts when using pyodbc.
#     """
    
#     odbc_connect = _build_odbc_connection_string()
#     connection_url = f"mssql+pyodbc:///?odbc_connect={odbc_connect}"
#     # Create engine; fast executemany helps when when using pandas. to_sql for large batches 
#     engine = create_engine(connection_url, fast_executemany = True, echo = echo)
#     return engine

# src/db.py (relevant parts)

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine

# load .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "1433")        # optional; can be left blank for named instance
DB_NAME = os.getenv("DB_NAME", "sales_analytics")
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
ODBC_DRIVER = os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server")

def _build_odbc_connection_string(use_port: bool = True) -> str:
    """
    Build and URL-encode the ODBC connection string for SQLAlchemy.
    - If use_port is False, we will omit the ",{DB_PORT}" after SERVER.
    """
    server_part = f"{DB_HOST},{DB_PORT}" if (DB_PORT and use_port) else f"{DB_HOST}"
    odbc_str = (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={server_part};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )
    # Return the URL-encoded string (for safe embedding in the SQLAlchemy URL)
    return quote_plus(odbc_str)

def get_engine(echo: bool = False, use_port: bool = True):
    """
    Create and return a SQLAlchemy engine.
    - echo: prints SQL statements when True (debug).
    - use_port: whether to include the port in SERVER (True -> localhost,1433).
    """
    odbc_connect = _build_odbc_connection_string(use_port=use_port)
    connection_url = f"mssql+pyodbc:///?odbc_connect={odbc_connect}"
    engine = create_engine(connection_url, fast_executemany=True, echo=echo)
    return engine
