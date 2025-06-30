import json
import ijson
from tqdm import tqdm
import os
import gc

# File paths (adjust as needed)
metadata_file = "amazon_parquet_data/amazon_metadata.json"
merged_file = "amazon_parquet_data/merged_users_all_final.json"
output_file = "amazon_parquet_data/merged_users_all_final_filtered.json"

# --- Streaming metadata load to avoid memory overflow ---
def load_valid_items(metadata_path):
    print("🔄 Streaming metadata file to extract valid ASINs...")
    valid_items = set()
    with open(metadata_path, 'rb') as f:
        for asin, _ in tqdm(ijson.kvitems(f, ''), desc="Loading ASINs", unit="asin"):
            valid_items.add(asin)
    print(f"✅ Loaded {len(valid_items):,} valid ASINs")
    return valid_items

# Load only ASIN keys from metadata in memory-safe way
valid_items = load_valid_items(metadata_file)

# --- Counters for analytics ---
cnt_total_events = 0
cnt_valid_events = 0

# --- Filtering logic ---
def filter_user_data(user_data):
    global cnt_total_events, cnt_valid_events
    history    = user_data.get("history", [])
    timestamps = user_data.get("timestamps", [])
    ratings    = user_data.get("ratings", [])
    reviews    = user_data.get("reviews", [])

    filtered = [
        (h, t, r, rev)
        for h, t, r, rev in zip(history, timestamps, ratings, reviews)
        if h in valid_items
    ]
    cnt_valid_events += len(filtered)
    cnt_total_events += len(history)

    if filtered:
        user_data["history"], user_data["timestamps"], user_data["ratings"], user_data["reviews"] = map(list, zip(*filtered))
    else:
        user_data["history"] = user_data["timestamps"] = user_data["ratings"] = user_data["reviews"] = []

    return user_data

# --- Main filtering loop ---
print("🔄 Processing merged JSON file and filtering records...")
total_size = os.path.getsize(merged_file)
processed_bytes = 0

with open(merged_file, 'rb') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write("{\n")
    first_record = True
    pbar = tqdm(total=total_size, desc="Filtering records", unit="byte")
    parser = ijson.kvitems(infile, "")

    record_count = 0
    for user_id, user_data in parser:
        filtered_data = filter_user_data(user_data)

        if not first_record:
            outfile.write(",\n")
        else:
            first_record = False

        json_record = json.dumps({user_id: filtered_data})[1:-1]  # remove outer braces
        outfile.write(json_record)

        record_count += 1
        current_pos = infile.tell()
        pbar.update(current_pos - processed_bytes)
        processed_bytes = current_pos

    pbar.close()
    outfile.write("\n}\n")

print(f"✅ Completed filtering {record_count} user records.")
print(f"✅ Output saved to: {output_file}")
print(f"📊 Total events: {cnt_total_events:,}")
print(f"✅ Valid events: {cnt_valid_events:,}")
print(f"❌ Filtered out {cnt_total_events - cnt_valid_events:,} invalid events.")