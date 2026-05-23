import sqlite3
import pandas as pd

# 1. Connect to the database
conn = sqlite3.connect('ml_monitor.db')
cursor = conn.cursor()

# 2. Ask SQLite to tell us the names of all existing tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("--- TABLES FOUND IN YOUR DATABASE ---")
if not tables:
    print("❌ No tables found! Your database is completely empty.")
    print("👉 Fix: Make sure to run your simulation script first (e.g., python3 examples/simulate_drift.py)")
else:
    for table in tables:
        table_name = table[0]
        print(f"\n📦 Table Name: '{table_name}'")
        print("-" * 30)
        
        # Read and print the first 5 rows of this specific table
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5;", conn)
            if df.empty:
                print("   (This table exists but has 0 rows of data)")
            else:
                print(df.to_string(index=False))
        except Exception as e:
            print(f"   Could not read table: {e}")

conn.close()