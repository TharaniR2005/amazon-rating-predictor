import sqlite3

connection = sqlite3.connect("amazon_rating.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT,
    review TEXT,
    rating REAL
)
""")

connection.commit()

print("Database created successfully!")

connection.close()