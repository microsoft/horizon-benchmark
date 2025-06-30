import pyarrow.parquet as pq
import pandas as pd
import logging
import argparse
from tqdm import tqdm
from itertools import chain
import os

output_dir = "amazon_parquet_data"
batch_size = 5000
list_columns = ["ratings", "timestamps", "history", "reviews"]

# Logging setup
logging.basicConfig(level=logging.INFO, filename='merging_user_histories_phase_two.log', filemode='w')
logger = logging.getLogger(__name__)

def read_parquet_batchwise(parquet_file):
    """Reads a Parquet file in small batches to prevent OOM issues."""
    try:
        parquet_reader = pq.ParquetFile(parquet_file)
        for batch in parquet_reader.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()
            for col in list_columns:
                df_batch[col] = df_batch[col].apply(lambda x: x if isinstance(x, list) else x)
            yield df_batch
    except Exception as e:
        print(f"⚠️ Skipping file {parquet_file}: {e}")

def merge_users(parquet_files):
    """Merges user histories efficiently without memory overhead."""
    user_data = {}  # Dictionary to hold aggregated user data

    for file in tqdm(parquet_files, desc="Processing Parquet Files"):
        print(f"Processing file: {file}")
        
        for batch in tqdm(read_parquet_batchwise(file), desc="Reading batches"):
            for _, row in batch.iterrows():
                user_id = row["user_id"]
                if user_id not in user_data:
                    user_data[user_id] = {"timestamps": [], "ratings": [], "history": [], "reviews": []}

                user_data[user_id]["timestamps"].extend(row["timestamps"])
                user_data[user_id]["ratings"].extend(row["ratings"])
                user_data[user_id]["history"].extend(row["history"])
                user_data[user_id]["reviews"].extend(row["reviews"])

    # Log one entry randomly to check the data
    if user_data:
        random_user_id = list(user_data.keys())[0]
        logger.info(f"Sample user data for user_id {random_user_id}: {user_data[random_user_id]}")

    # Convert dictionary to DataFrame
    print("Sorting and converting user data to DataFrame...")
    grouped_users = []
    for user_id, data in tqdm(user_data.items(), desc="Sorting user histories"):
        sorted_data = sorted(zip(data["timestamps"], data["ratings"], data["history"], data["reviews"])) if data["timestamps"] else []
        
        if sorted_data:
            timestamps, ratings, history, reviews = zip(*sorted_data)
        else:
            timestamps, ratings, history, reviews = [], [], [], []

        grouped_users.append({
            "user_id": user_id,
            "timestamps": list(timestamps),
            "ratings": list(ratings),
            "history": list(history),
            "reviews": list(reviews)
        })
    
    # Log one entry randomly to check the data
    if grouped_users:
        random_user = grouped_users[0]
        logger.info(f"Sample user data for user_id {random_user['user_id']}: {random_user}")

    return pd.DataFrame(grouped_users)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Merge user histories from Parquet files.")
    parser.add_argument("--input_file_one", type=str, default="test1.parquet", help="First input Parquet file.")
    parser.add_argument("--input_file_two", type=str, default="test2.parquet", help="Second input Parquet file.")
    parser.add_argument("--output_file", type=str, default="merged_users_train_[0-10].parquet", help="Output Parquet file.")
    args = parser.parse_args()
    input_file_one = args.input_file_one
    input_file_two = args.input_file_two
    output_file = args.output_file

    parquet_files = [f"{output_dir}/{input_file_one}", f"{output_dir}/{input_file_two}"]

    # Process the Parquet files
    merged_df = merge_users(parquet_files)

    if merged_df.empty:
        print("❌ No valid data to process.")
        exit()

    output_path = f"{output_dir}/{output_file}"
    merged_df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    print(f"✅ Merged Parquet file saved at {output_path}")
    print(f"Number of unique users: {len(merged_df)}")
    print(f"Number of records: {merged_df.shape[0]}")
    logger.info(f"Merged Parquet file saved at {output_path}")
    logger.info(f"Number of unique users: {len(merged_df)}")
    logger.info(f"Number of records: {merged_df.shape[0]}")
    print("✅ Merging user histories completed successfully.")
