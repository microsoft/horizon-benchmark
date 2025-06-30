from datasets import load_dataset
import pandas as pd
import json
import os
from tqdm import tqdm
import pyarrow.parquet as pq

output_dir = "amazon_parquet_data"
os.makedirs(output_dir, exist_ok=True)
# Read category names
with open("category_names.txt", "r") as file:
    category_names_main = [line.strip() for line in file]

print(f"Processing categories: {category_names_main}")
print(f"Total categories: {len(category_names_main)}")

master_dict = {}

for category in category_names_main:
    print(f"🔄 Processing category: {category}")
    try:
        dataset_metadata = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True
        )
        df = dataset_metadata.to_pandas()[["main_category", "title", "features", "parent_asin"]]
        df = df.dropna(subset=["parent_asin"])

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {category}"):
            
            # Ensure all list elements are strings
            row["features"] = [str(feature) for feature in row["features"]]
            row["title"] = str(row["title"])
            row["main_category"] = str(row["main_category"])
            row["parent_asin"] = str(row["parent_asin"])
            parent_asin = row["parent_asin"]
            master_dict[parent_asin] = {
                "main_category": row["main_category"],
                "title": row["title"],
                "features": row["features"]
            }

    except Exception as e:
        print(f"❌ Failed to process {category}: {e}")

# Save to JSON
json_path = os.path.join(output_dir, "amazon_metadata.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(master_dict, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved JSON file with {len(master_dict)} items")

# Save to Parquet
df_all = pd.DataFrame.from_dict(master_dict, orient="index").reset_index()
df_all = df_all.rename(columns={"index": "parent_asin"})
parquet_path = os.path.join(output_dir, "amazon_metadata.parquet")
df_all.to_parquet(parquet_path, index=False)

print(f"✅ Done! Saved {len(master_dict)} items")
print(f"📦 JSON: {json_path}")
print(f"📦 Parquet: {parquet_path}")
