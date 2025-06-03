import sqlite3

# Connect to the database
conn = sqlite3.connect("/home/hpmcclea/SemiF-SyntheticPipeline/data/db/agir.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in agir.db:")
for table in tables:
    print(table[0])

# Pick a table and show some rows (replace 'your_table' with an actual table name)
table_name = tables[0][0]  # Example: pick first table
print(f"\nSample data from table '{table_name}':")
cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Optionally: show column names
cursor.execute(f"PRAGMA table_info({table_name});")
columns = cursor.fetchall()
print("\nColumns:")
for col in columns:
    print(col[1])  # col[1] is the column name

conn.close()