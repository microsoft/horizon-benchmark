import pandas as pd
import pyarrow.parquet as pq
import json
from tqdm import tqdm
import numpy as np
import os

# Directory where Parquet files are saved
output_dir = "amazon_parquet_data"
category = "merged_users_all_final"
parquet_filename = 'amazon_parquet_data/merged_users_all_final.parquet'

# Load Parquet file batch-wise (to avoid memory issues)
parquet_file = pq.ParquetFile(parquet_filename)

df_list = []
for batch in tqdm(parquet_file.iter_batches(batch_size=10000)):  # Adjust batch size
    df_batch = batch.to_pandas()

    # Convert list columns to JSON strings (so Pandas can handle them)
    for col in ["ratings", "timestamps", "history", "reviews"]:  
        df_batch[col] = df_batch[col].apply(lambda x: x if isinstance(x, list) else x)

    df_list.append(df_batch)

# Combine all batches into a full DataFrame
df = pd.concat(df_list, ignore_index=True)

# Print length of the DataFrame
print(f"Number of users: {len(df)}")

# Prepare for dumping df to a JSON
data = {}
for row in tqdm(df.itertuples(index=False), total=len(df)):
    user_id = row[0]

    # Convert all possible NumPy arrays to Python lists
    ratings = row[2].tolist() if isinstance(row[2], (list, pd.Series, np.ndarray)) else list(row[2])
    timestamps = row[1].tolist() if isinstance(row[1], (list, pd.Series, np.ndarray)) else list(row[1])
    history = row[3].tolist() if isinstance(row[3], (list, pd.Series, np.ndarray)) else list(row[3])
    reviews = row[4].tolist() if isinstance(row[4], (list, pd.Series, np.ndarray)) else list(row[4])

    # Store the structured data
    data[user_id] = {
        "ratings": ratings,
        "timestamps": timestamps,
        "history": history,
        "reviews": reviews
    }

# Dump to JSON file
output_json_file = os.path.join(output_dir, f"{category}.json")
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)