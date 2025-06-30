import sqlite3
import ijson
from tqdm import tqdm

# Input and output paths
json_path = "amazon_parquet_data/amazon_metadata.json"
sqlite_path = "amazon_parquet_data/metadata_titles.db"

# SQLite setup
conn = sqlite3.connect(sqlite_path)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS metadata (
    asin TEXT PRIMARY KEY,
    title TEXT,
    category TEXT
)
""")
conn.commit()

# Stream JSON using ijson
print("🔄 Streaming and inserting JSON records into SQLite...")
batch_size = 10000
batch = []
total = 0

with open(json_path, "rb") as f:
    parser = ijson.kvitems(f, "")  # parse key-value pairs at top-level

    for asin, data in tqdm(parser, desc="Inserting", unit=" items"):
        title = str(data.get("title", ""))
        category = str(data.get("main_category", ""))
        batch.append((asin, title, category))
        total += 1

        if len(batch) >= batch_size:
            cursor.executemany("INSERT OR REPLACE INTO metadata (asin, title, category) VALUES (?, ?, ?)", batch)
            conn.commit()
            batch = []

# Final flush
if batch:
    cursor.executemany("INSERT OR REPLACE INTO metadata (asin, title, category) VALUES (?, ?, ?)", batch)
    conn.commit()

conn.close()
print(f"✅ Done! Inserted {total} items into {sqlite_path}")