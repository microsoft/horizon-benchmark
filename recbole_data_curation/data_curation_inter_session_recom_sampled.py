import os
import time
import random
from datetime import datetime, timezone
from tqdm import tqdm
import ijson
import json

# --- Configuration ---
SEED = 42
random.seed(SEED)

MASTER_JSON_PATH = "amazon_parquet_data/merged_users_all_final_filtered_no_reviews.json"
SPLIT_DIR = "splits"
OUTPUT_DIR = "recbole_dataset_sampled"  # Output directory for .inter files
IND_VAL_USERS_FILE = os.path.join(SPLIT_DIR, "ind_val_users.json")
IND_TEST_USERS_FILE = os.path.join(SPLIT_DIR, "ind_test_users.json")
OOD_VAL_USERS_FILE = os.path.join(SPLIT_DIR, "ood_val_users.json")
OOD_TEST_USERS_FILE = os.path.join(SPLIT_DIR, "ood_test_users.json")

TS_2019_START = 2019
TS_2020_START = 2020

HEADER_SEQ = "session_id:token\titem_id_list:token_seq\titem_id:token\n"

SAMPLE_SIZE_PER_GROUP = 25000
IND_TRAIN_TOTAL = 150000

# Load user splits
def load_user_ids(filepath):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

# Load user ids json
def load_user_ids_json(filepath):
    # Open the json and load the dict
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Filter out user_ids where the length of ratings is less than 2
    data = [user_id for user_id, user_data in data.items() if len(user_data.get('history', [])) >= 5]
    return data

def save_user_ids(filepath, user_ids):
    with open(filepath, 'w') as f:
        for uid in sorted(user_ids):
            f.write(f"{uid}\n")

def get_year(ts):
    return int(datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y'))

# Main processing
def process_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load full user ID jsons
    ind_val_all = load_user_ids_json(IND_VAL_USERS_FILE)
    print(f"ind_val_all: {len(ind_val_all)}")
    ind_test_all = load_user_ids_json(IND_TEST_USERS_FILE)
    print(f"ind_test_all: {len(ind_test_all)}")
    ood_val_all = load_user_ids_json(OOD_VAL_USERS_FILE)
    print(f"ood_val_all: {len(ood_val_all)}")
    ood_test_all = load_user_ids_json(OOD_TEST_USERS_FILE)
    print(f"ood_test_all: {len(ood_test_all)}")
    all_users = load_user_ids_json(MASTER_JSON_PATH)
    print(f"all_users: {len(all_users)}")

    # Remove users which are in ind_val_all, ind_test_all, ood_val_all, ood_test_all from all_users
    all_users = set(all_users) - (set(ind_val_all) | set(ind_test_all) | set(ood_val_all) | set(ood_test_all))
    all_users = list(all_users)
    print(f"all_users (after removing): {len(all_users)}")

    # Sample 25k from each split
    ind_val_users = set(random.sample(ind_val_all, SAMPLE_SIZE_PER_GROUP))
    ind_test_users = set(random.sample(ind_test_all, SAMPLE_SIZE_PER_GROUP))
    ood_val_users = set(random.sample(ood_val_all, SAMPLE_SIZE_PER_GROUP))
    ood_test_users = set(random.sample(ood_test_all, SAMPLE_SIZE_PER_GROUP))

    # Sample additional 50k for ind_train
    additional_train_users = set(random.sample(all_users, IND_TRAIN_TOTAL - SAMPLE_SIZE_PER_GROUP * 2))
    ind_train_users = ind_val_users | ind_test_users | additional_train_users

    # Save sampled user IDs to new txt files
    save_user_ids(os.path.join(SPLIT_DIR, "ind_train_user_ids_sampled.txt"), ind_train_users)
    save_user_ids(os.path.join(SPLIT_DIR, "ind_val_user_ids_sampled.txt"), ind_val_users)
    save_user_ids(os.path.join(SPLIT_DIR, "ind_test_user_ids_sampled.txt"), ind_test_users)
    save_user_ids(os.path.join(SPLIT_DIR, "ood_val_user_ids_sampled.txt"), ood_val_users)
    save_user_ids(os.path.join(SPLIT_DIR, "ood_test_user_ids_sampled.txt"), ood_test_users)

    print("Sampled user counts:")
    print(f"  ind_train: {len(ind_train_users)}")
    print(f"  ind_val:   {len(ind_val_users)}")
    print(f"  ind_test:  {len(ind_test_users)}")
    print(f"  ood_val:   {len(ood_val_users)}")
    print(f"  ood_test:  {len(ood_test_users)}")

    # Open six .inter files
    files = {}
    for split in ["train_ind", "val_ind", "test_ind", "train_ood", "val_ood", "test_ood"]:
        path = os.path.join(OUTPUT_DIR, f"{OUTPUT_DIR}.{split}_sampled.inter")
        f = open(path, 'w')
        f.write(HEADER_SEQ)
        files[split] = f
        print(f"Writing to: {path}")

    # Stats
    stats = {k: 0 for k in ['total_users', 'skipped_short', 'skipped_mismatch', 'processed_users',
                            'train_ind', 'val_ind', 'test_ind', 'train_ood', 'val_ood', 'test_ood']}

    # Stream JSON
    # cnt_ind_train = 0
    # cnt_ind_val = 0
    # cnt_ind_test = 0
    # cnt_ood_val = 0
    # cnt_ood_test = 0
    with open(MASTER_JSON_PATH, 'rb') as f_json:
        parser = ijson.kvitems(f_json, '')
        for user_id, data in tqdm(parser, desc="Users", unit="user"):
            stats['total_users'] += 1

            # Filter to sampled users only
            if user_id not in ind_train_users and user_id not in ind_val_users and \
               user_id not in ind_test_users and user_id not in ood_val_users and \
               user_id not in ood_test_users:
                continue

            history = data.get('history', [])
            timestamps = data.get('timestamps', [])

            if len(history) < 2:
                stats['skipped_short'] += 1
                continue
            if len(history) != len(timestamps):
                stats['skipped_mismatch'] += 1
                continue

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

            # Flags
            is_ood_test = user_id in ood_test_users
            is_ood_val = user_id in ood_val_users
            is_ind_val = user_id in ind_val_users
            is_ind_test = user_id in ind_test_users
            is_ind_train = user_id in ind_train_users

            for i in range(1, len(interactions)):
                item_id, ts_i, year = interactions[i]
                prefix_items = [iid for iid, _, _ in interactions[:i]]
                line = f"{user_id}\t{' '.join(prefix_items)}\t{item_id}\n"

                if is_ood_test:
                    if year >= TS_2020_START:
                        files['test_ood'].write(line); stats['test_ood'] += 1
                    elif TS_2019_START <= year < TS_2020_START:
                        files['val_ood'].write(line); stats['val_ood'] += 1
                elif is_ood_val:
                    if TS_2019_START <= year < TS_2020_START:
                        files['val_ood'].write(line); stats['val_ood'] += 1
                elif is_ind_test:
                    if year >= TS_2020_START:
                        files['test_ind'].write(line); stats['test_ind'] += 1
                    elif TS_2019_START <= year < TS_2020_START:
                        files['val_ind'].write(line); stats['val_ind'] += 1
                    elif year < TS_2019_START:
                        files['train_ind'].write(line); stats['train_ind'] += 1
                elif is_ind_val:
                    if TS_2019_START <= year < TS_2020_START:
                        files['val_ind'].write(line); stats['val_ind'] += 1
                    elif year < TS_2019_START:
                        files['train_ind'].write(line); stats['train_ind'] += 1
                elif is_ind_train:
                    if year < TS_2019_START:
                        files['train_ind'].write(line); stats['train_ind'] += 1
                    # else:
                    #     # IND / other users
                    #     if year < TS_2019_START:
                    #         files['train_ind'].write(line); stats['train_ind'] += 1
                    #     elif TS_2019_START <= year < TS_2020_START:
                    #         if is_ind_val:
                    #             files['val_ind'].write(line); stats['val_ind'] += 1

                # else:
                #     # IND / other users
                #     if year < TS_2019_START:
                #         files['train_ind'].write(line); stats['train_ind'] += 1
                #     elif TS_2019_START <= year < TS_2020_START:
                #         if is_ind_val:
                #             files['val_ind'].write(line); stats['val_ind'] += 1
                #     else:
                #         if is_ind_test:
                #             files['test_ind'].write(line); stats['test_ind'] += 1


    for f in files.values():
        f.close()

    print("\n--- Processing Summary ---")
    print(f"Total users seen: {stats['total_users']}")
    print(f"Processed users: {stats['processed_users']}")
    print(f"Skipped (short): {stats['skipped_short']}")
    print(f"Skipped (mismatch): {stats['skipped_mismatch']}")
    print(f"IND Train: {stats['train_ind']}, Val: {stats['val_ind']}, Test: {stats['test_ind']}")
    print(f"OOD Train: {stats['train_ood']}, Val: {stats['val_ood']}, Test: {stats['test_ood']}")

if __name__ == "__main__":
    start = time.time()
    process_data()
    print(f"Total time: {time.time() - start:.2f}s")
