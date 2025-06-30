import os
import time
from datetime import datetime, timezone
from tqdm import tqdm
import ijson  # streaming parser for large JSON

# --- Configuration ---
MASTER_JSON_PATH = "amazon_parquet_data/merged_users_all_final_filtered_no_reviews.json"
SPLIT_DIR = "splits"
OUTPUT_DIR = "recbole_dataset_streamed"  # Output directory for .inter files
IND_VAL_USERS_FILE = os.path.join(SPLIT_DIR, "ind_val_user_ids.txt")
IND_TEST_USERS_FILE = os.path.join(SPLIT_DIR, "ind_test_user_ids.txt")
OOD_VAL_USERS_FILE = os.path.join(SPLIT_DIR, "ood_val_user_ids.txt")
OOD_TEST_USERS_FILE = os.path.join(SPLIT_DIR, "ood_test_user_ids.txt")

# Time boundaries by year
TS_2019_START = 2019
TS_2020_START = 2020

# RecBole sequential .inter header (session_id, item_id_list, item_id)
# Tab-separated columns; item_id_list tokens space-separated
HEADER_SEQ = "session_id:token\titem_id_list:token_seq\titem_id:token\n"

# Load user splits
def load_user_ids(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath) as f:
        return {line.strip() for line in f if line.strip()}

# Determine year from millisecond timestamp
def get_year(ts):
    return int(datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y'))

# Main processing
def process_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load splits
    ind_val = load_user_ids(IND_VAL_USERS_FILE)
    ind_test = load_user_ids(IND_TEST_USERS_FILE)
    ood_val = load_user_ids(OOD_VAL_USERS_FILE)
    ood_test = load_user_ids(OOD_TEST_USERS_FILE)

    # Open six files and write headers
    files = {}
    for split in [f"train_ind", f"val_ind", "test_ind", "train_ood", "val_ood", "test_ood"]:
        path = os.path.join(OUTPUT_DIR, f"{OUTPUT_DIR}.{split}.inter")
        f = open(path, 'w')
        f.write(HEADER_SEQ)
        files[split] = f

    # Initialize stats
    stats = {k: 0 for k in ['total_users','skipped_short','skipped_mismatch','processed_users',
                              'train_ind','val_ind','test_ind','train_ood','val_ood','test_ood']}

    # Stream JSON per user with tqdm
    with open(MASTER_JSON_PATH, 'rb') as f_json:
        parser = ijson.kvitems(f_json, '')
        for user_id, data in tqdm(parser, desc="Users", unit="user"):
            stats['total_users'] += 1
            history = data.get('history', [])
            timestamps = data.get('timestamps', [])

            # Skip invalid users
            if len(history) < 2:
                stats['skipped_short'] += 1
                continue
            if len(history) != len(timestamps):
                stats['skipped_mismatch'] += 1
                continue

            # Build sorted interactions list: (item_id, ts, year)
            interactions = []
            for item, ts in zip(history, timestamps):
                try:
                    ts_i = int(ts)
                    year = get_year(ts_i)
                    interactions.append((str(item), ts_i, year))
                except:
                    continue
            if len(interactions) < 2:
                stats['skipped_short'] += 1
                continue
            interactions.sort(key=lambda x: x[1])
            stats['processed_users'] += 1

            # Precompute flags
            is_ood_test = user_id in ood_test
            is_ood_val = user_id in ood_val
            is_ind_val = user_id in ind_val
            is_ind_test = user_id in ind_test

            # Generate prefix-target pairs with correct splitting logic
            # Skip index 0, start from 1
            for i in range(1, len(interactions)):
                item_id, ts_i, year = interactions[i]
                # prefix items up to i
                prefix_items = [iid for iid, _, _ in interactions[:i]]
                line = f"{user_id}\t{' '.join(prefix_items)}\t{item_id}\n"

                # OOD logic first
                if is_ood_test:
                    if year >= TS_2020_START:
                        files['test_ood'].write(line); stats['test_ood'] += 1
                    elif TS_2019_START <= year < TS_2020_START:
                        files['val_ood'].write(line); stats['val_ood'] += 1
                    else:
                        files['train_ood'].write(line); stats['train_ood'] += 1
                elif is_ood_val:
                    if TS_2019_START <= year < TS_2020_START:
                        files['val_ood'].write(line); stats['val_ood'] += 1
                    elif year < TS_2019_START:
                        files['train_ood'].write(line); stats['train_ood'] += 1
                    # ignore post-2019 for OOD val
                else:
                    # IND / other users
                    if year < TS_2019_START:
                        files['train_ind'].write(line); stats['train_ind'] += 1
                    elif TS_2019_START <= year < TS_2020_START:
                        if is_ind_val:
                            files['val_ind'].write(line); stats['val_ind'] += 1
                    else:
                        if is_ind_test:
                            files['test_ind'].write(line); stats['test_ind'] += 1

    # Close files
    for f in files.values(): f.close()

    # Print summary
    print("--- Processing Summary ---")
    print(f"Total users: {stats['total_users']}")
    print(f"Processed users: {stats['processed_users']}")
    print(f"Skipped (short): {stats['skipped_short']}")
    print(f"Skipped (mismatch): {stats['skipped_mismatch']}")
    print(f"IND Train: {stats['train_ind']}, Val: {stats['val_ind']}, Test: {stats['test_ind']}")
    print(f"OOD Train: {stats['train_ood']}, Val: {stats['val_ood']}, Test: {stats['test_ood']}")

if __name__ == "__main__":
    start = time.time()
    process_data()
    print(f"Total time: {time.time() - start:.2f}s")
