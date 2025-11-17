# scripts/pyodbc_test.py
import pyodbc

drv = "ODBC Driver 17 for SQL Server"
host_port = "localhost,1433"
host_only = "localhost"

conn1 = f"DRIVER={{{drv}}};SERVER={host_port};DATABASE=sales_analytics;UID=sa;PWD=StrongPassword123#;Encrypt=no;"
conn2 = f"DRIVER={{{drv}}};SERVER={host_only};DATABASE=sales_analytics;UID=sa;PWD=StrongPassword123#;Encrypt=no;"

def try_connect(conn_str, name):
    print(f"\nTrying {name}:")
    print(conn_str)
    try:
        cn = pyodbc.connect(conn_str, timeout=5)
        print("-> ✅ SUCCESS:", cn)
        cn.close()
    except Exception as e:
        print("-> ❌ ERROR:", repr(e))

if __name__ == "__main__":
    try_connect(conn1, "conn1 (localhost,1433)")
    try_connect(conn2, "conn2 (localhost)")
