import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
import logging
import datasets
from datasets import load_dataset
datasets.logging.set_verbosity_error()

# Output Directory for Parquet files
output_dir = "amazon_parquet_data"
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load dataset (without shared memory)
def load_reviews_dataset(category):
    # If parquet file already exists, skip loading
    if os.path.exists(os.path.join(output_dir, f"{category}_index.parquet")):
        print(f"⚠️ {category} index already exists, skipping...")
        return

    print(f"Loading {category} dataset...")
    # Load the dataset
    dataset_review_samples = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                                          f"raw_review_{category}", 
                                          trust_remote_code=True)["full"]

    user_index = {}
    for item in tqdm(dataset_review_samples, desc=f"Setting up {category} dataset user index", unit="item"):
        user_id = item["user_id"]
        timestamp = item["timestamp"]

        if user_id not in user_index:
            user_index[user_id] = {}
        user_index[user_id][timestamp] = f"{item['title']} \n{item['text']}"

    # Save as JSON to avoid shared dict issues
    save_path = os.path.join(output_dir, f"{category}_index.parquet")
    df = pd.DataFrame([(user, ts, review) for user, timestamps in user_index.items() for ts, review in timestamps.items()],
                      columns=["user_id", "timestamp", "review"])
    df.to_parquet(save_path, engine="pyarrow", compression="snappy")

    print(f"✅ Saved {category} index to {save_path}")

# Function to process each category
def process_category(category):
    print(f"Processing {category}...")

    # If processed Parquet file already exists, skip processing
    if os.path.exists(os.path.join(output_dir, f"{category}.parquet")):
        print(f"⚠️ {category} already processed, skipping...")
        return

    # Load previously saved dataset from Parquet
    review_index_path = os.path.join(output_dir, f"{category}_index.parquet")
    if not os.path.exists(review_index_path):
        print(f"⚠️ Missing {review_index_path}, skipping category {category}")
        return

    review_index = pd.read_parquet(review_index_path).set_index(["user_id", "timestamp"])

    # Load the dataset with history
    dataset_timestamp_his_train = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                                            f"0core_timestamp_w_his_{category}", 
                                            trust_remote_code=True)["train"]
    dataset_timestamp_his_val = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                                         f"0core_timestamp_w_his_{category}", 
                                         trust_remote_code=True)["valid"]
    dataset_timestamp_his_test = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                                         f"0core_timestamp_w_his_{category}", 
                                         trust_remote_code=True)["test"]
    
    # Convert to DataFrame
    dataset_timestamp_his_train = pd.DataFrame(dataset_timestamp_his_train)
    dataset_timestamp_his_val = pd.DataFrame(dataset_timestamp_his_val)
    dataset_timestamp_his_test = pd.DataFrame(dataset_timestamp_his_test)
    # Merge validation and test datasets
    dataset_timestamp_his = pd.concat([dataset_timestamp_his_val, dataset_timestamp_his_test], ignore_index=True)
    dataset_timestamp_his = pd.concat([dataset_timestamp_his_train, dataset_timestamp_his], ignore_index=True)

    user_data = {}
    for _, item in tqdm(dataset_timestamp_his.iterrows(), desc=f"Processing {category} dataset", unit="item"):
        user_id = item["user_id"]
        timestamp = item["timestamp"]
        user_id_str = str(user_id)
        timestamp = int(timestamp)

        if user_id_str not in user_data:
            user_data[user_id_str] = {
                "user_id": user_id_str,
                "ratings": [],
                "timestamps": [],
                "history": [],
                "reviews": []
            }

        user_data[user_id_str]["ratings"].append(item["rating"])
        user_data[user_id_str]["timestamps"].append(item["timestamp"])
        user_data[user_id_str]["history"].append(item["parent_asin"])

        # Lookup review text from pre-saved index
        review_text = "NA"
        try:
            review_text = review_index.loc[(user_id_str, timestamp), "review"]
        except KeyError:
            print(f"⚠️ Review not found for user {user_id_str} at timestamp {timestamp}")

        user_data[user_id_str]["reviews"].append(review_text)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(user_data, orient="index")

    # Save as Parquet
    parquet_filename = os.path.join(output_dir, f"{category}.parquet")
    df.to_parquet(parquet_filename, engine="pyarrow", compression="snappy")

    print(f"✅ Saved {category} to {parquet_filename}")
    logging.info(f"Saved {category} to {parquet_filename}")
    logging.info(f"Number of users in {category}: {len(df)}")

if __name__ == "__main__":
    # Read category names from file
    category_names = []
    with open("category_names.txt", "r") as file:
        for line in file:
            category_names.append(line.strip())
    print(category_names)

    # Parallel execution for dataset loading
    with multiprocessing.Pool(processes=10) as pool:
        list(tqdm(pool.imap(load_reviews_dataset, category_names), total=len(category_names)))

    print("✅ All category indexes saved!")

    # Process each category in parallel
    with multiprocessing.Pool(processes=10) as pool:
        list(tqdm(pool.imap(process_category, category_names), total=len(category_names)))

    print("✅ All categories processed and saved as Parquet!")

    # Clean up
    for category in category_names:
        review_index_path = os.path.join(output_dir, f"{category}_index.parquet")
        if os.path.exists(review_index_path):
            os.remove(review_index_path)
            print(f"🗑️ Removed {review_index_path}")

    print("✅ Cleanup complete!")

    # Final message
    print("All datasets have been processed and saved successfully!")
    logging.info("All datasets have been processed and saved successfully!")
    print("Check the 'amazon_parquet_data' directory for the output files.")
    logging.info("Check the 'amazon_parquet_data' directory for the output files.")
# End of script
# Note: Make sure to have the required libraries installed and the dataset available.
# This script is designed to process Amazon review datasets and save them in Parquet format.