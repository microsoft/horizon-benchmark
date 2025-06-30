import ijson
import json
from tqdm import tqdm

# === CONFIGURATION ===
input_file = "amazon_parquet_data/filtered_users_all_final_filtered_no_reviews.json"

split_config = {
    "ood_val": {
        "ids_file": "splits/ood_val_user_ids.txt",
        "json_file": "splits/ood_val_users.json"
    },
    "ood_test": {
        "ids_file": "splits/ood_test_user_ids.txt",
        "json_file": "splits/ood_test_users.json"
    },
    "ind_val": {
        "ids_file": "splits/ind_val_user_ids.txt",
        "json_file": "splits/ind_val_users.json"
    },
    "ind_test": {
        "ids_file": "splits/ind_test_user_ids.txt",
        "json_file": "splits/ind_test_users.json"
    }
}

# === LOAD USER ID SETS ===
print("📂 Loading user ID sets from .txt files...")
split_user_ids = {}
for split_name, paths in split_config.items():
    with open(paths["ids_file"], 'r') as f:
        split_user_ids[split_name] = set(line.strip() for line in f)
    print(f"✅ Loaded {len(split_user_ids[split_name]):,} users for {split_name}")

# === OPEN JSON WRITERS FOR EACH SPLIT ===
writers = {}
first_flags = {}
for split_name, paths in split_config.items():
    f = open(paths["json_file"], 'w', encoding='utf-8')
    f.write('{\n')
    writers[split_name] = f
    first_flags[split_name] = True

# === MAIN PASS THROUGH INPUT FILE ===
print("🔄 Scanning and writing users in a single pass...")
with open(input_file, 'rb') as f:
    for user_id, user_data in tqdm(ijson.kvitems(f, ''), desc="Processing users"):
        for split_name, id_set in split_user_ids.items():
            if user_id in id_set:
                if not first_flags[split_name]:
                    writers[split_name].write(',\n')
                first_flags[split_name] = False
                json.dump(user_id, writers[split_name])
                writers[split_name].write(': ')
                json.dump(user_data, writers[split_name])
                break  # no need to check other splits

# === FINALIZE OUTPUT FILES ===
for split_name, f in writers.items():
    f.write('\n}\n')
    f.close()
    print(f"📦 Saved {split_name} to {split_config[split_name]['json_file']}")

print("✅ All splits written successfully!")
