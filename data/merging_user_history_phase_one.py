import pandas as pd
import glob
import pyarrow.parquet as pq
import json
from tqdm import tqdm
from itertools import chain
import argparse
import os
import sys
import logging

output_dir = "amazon_parquet_data"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='merging_user_histories.log', filemode='w')
logger = logging.getLogger(__name__)

def enforce_string_dtype(df, list_columns):
    """Ensure all elements in list columns are strings."""
    for col in list_columns:
        df[col] = df[col].apply(lambda x: [str(i) for i in x] if isinstance(x, list) else [])
    return df

def read_parquet_batchwise(parquet_file, batch_size=10000):
    """Reads a Parquet file in small batches and ensures list columns remain lists."""
    print(f"Reading Parquet file: {parquet_file}")
    try:
        parquet_reader = pq.ParquetFile(parquet_file)
        df_list = []
        for batch in parquet_reader.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()

            # Ensure these columns contain lists, not NaN or strings
            for col in ["ratings", "timestamps", "history", "reviews"]:
                df_batch[col] = df_batch[col].apply(lambda x: x if isinstance(x, list) else x)

            df_list.append(df_batch)

        print(f"✅ Successfully read {len(df_list)} batches from {parquet_file}")
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Skipping corrupted file: {parquet_file} (Error: {e})")
        return pd.DataFrame()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Merge user histories from Parquet files.")
    parser.add_argument("--start", type=int, default=0, help="Start index for categories.")
    parser.add_argument("--end", type=int, default=5, help="End index for categories.")
    args = parser.parse_args()
    start_index = args.start
    end_index = args.end

    # Read category names
    with open("category_names.txt", "r") as file:
        category_names_main = [line.strip() for line in file]

    category_names = category_names_main[start_index:end_index]
    print(f"Processing categories: {category_names}")
    parquet_files = [f"{output_dir}/{category}.parquet" for category in category_names]

    list_columns = ["ratings", "timestamps", "history", "reviews"]
    all_data = []

    # Read Parquet files sequentially
    for file in tqdm(parquet_files, desc="Reading Parquet Files"):
        df = read_parquet_batchwise(file)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("❌ No valid data to process.")
        exit()

    # Concatenate all DataFrames
    merged_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Concatenated DataFrames: {merged_df.shape}")
    print(merged_df)

    # Group by user_id
    print("Aggregating data by user_id...")
    grouped = []
    for user_id, group in tqdm(merged_df.groupby("user_id"), desc="Aggregating by user_id"):
        timestamps = list(chain.from_iterable(group["timestamps"].dropna()))
        ratings = list(chain.from_iterable(group["ratings"].dropna()))
        history = list(chain.from_iterable(group["history"].dropna()))
        reviews = list(chain.from_iterable(group["reviews"].dropna()))

        # ✅ Sort all data by timestamps
        sorted_data = sorted(zip(timestamps, ratings, history, reviews))

        if sorted_data:
            timestamps, ratings, history, reviews = zip(*sorted_data)
        else:
            timestamps, ratings, history, reviews = [], [], [], []

        aggregated = {
            "user_id": user_id,
            "timestamps": list(timestamps),
            "ratings": list(ratings),
            "history": list(history),
            "reviews": list(reviews)
        }
        grouped.append(aggregated)

    merged_df = pd.DataFrame(grouped)
    

    # Ensure all list elements are strings
    merged_df = enforce_string_dtype(merged_df, list_columns)

    # Save final dataset
    output_path = f"{output_dir}/merged_users_[{start_index}-{end_index}].parquet"
    merged_df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    print(f"✅ Merged Parquet file saved at {output_path}")
    print(f"Number of unique users: {len(merged_df)}")
    print(f"Number of records: {merged_df.shape[0]}")
    logger.info(f"Merged Parquet file saved at {output_path}")
    logger.info(f"Number of unique users: {len(merged_df)}")
    logger.info(f"Number of records: {merged_df.shape[0]}")
    print("✅ Merging user histories completed successfully.")