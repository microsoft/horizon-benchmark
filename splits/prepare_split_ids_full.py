import ijson
import json
import random
from datetime import datetime, timezone
from tqdm import tqdm

# === CONFIGURATION ===
input_file = "amazon_parquet_data/merged_users_all_final_filtered_no_reviews.json"
val_output_file = "splits/ood_val_users.json"
test_output_file = "splits/ood_test_users.json"
val_ids_file = "splits/ood_val_user_ids.txt"
test_ids_file = "splits/ood_test_user_ids.txt"
val_user_limit = 1_000_000
test_user_limit = 1_000_000
TEST_YEAR_THRESHOLD = 2020
VAL_YEAR_THRESHOLD = 2019
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# === HELPERS ===
def get_year(ts):
    return int(datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime('%Y'))

# === PASS 1: Identify eligible user IDs ===
ood_val_candidates = []
ood_test_candidates = []

print("🔍 First pass: Collecting eligible user IDs")
with open(input_file, 'rb') as f:
    for user_id, user_data in tqdm(ijson.kvitems(f, ''), desc="Scanning users for OOD candidates"):
        timestamps_lst = user_data.get("timestamps", [])
        if not timestamps_lst:
            continue

        timestamps_lst_reversed = list(reversed(timestamps_lst))

        for timestamp in timestamps_lst_reversed:
            timestamp = int(timestamp)
            # Get the first timestamp in the original order
            first_timestamp = int(timestamps_lst[0])
            first_year = datetime.fromtimestamp(first_timestamp / 1000, tz=timezone.utc).strftime('%Y')
            year = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y')
            first_year = int(first_year)
            year = int(year)
            if year == TEST_YEAR_THRESHOLD:
                ood_test_candidates.append(user_id)
                if first_year <= VAL_YEAR_THRESHOLD:
                    ood_val_candidates.append(user_id)
                break
            elif year == VAL_YEAR_THRESHOLD:
                ood_val_candidates.append(user_id)
                break

print(f"✅ Found {len(ood_test_candidates):,} test candidates and {len(ood_val_candidates):,} val candidates")

# === Randomly sample 1M from each ===
ood_test_sampled = set(random.sample(ood_test_candidates, min(test_user_limit, len(ood_test_candidates))))
# remove test users from val candidates
ood_val_candidates = [user_id for user_id in ood_val_candidates if user_id not in ood_test_sampled]
ood_val_sampled = set(random.sample(ood_val_candidates, min(val_user_limit, len(ood_val_candidates))))
print(f"✅ Sampled {len(ood_test_sampled):,} test users and {len(ood_val_sampled):,} val users")

# === Save user IDs to text files ===
print("💾 Saving user IDs to text files")
# Save user ID files
with open(test_ids_file, 'w') as f_test_ids:
    f_test_ids.writelines(user_id + '\n' for user_id in ood_test_sampled)

with open(val_ids_file, 'w') as f_val_ids:
    f_val_ids.writelines(user_id + '\n' for user_id in ood_val_sampled)

# === PASS 2: Write user data to JSON files ===
print("💾 Second pass: Writing selected users to JSON")

test_json = open(test_output_file, 'w', encoding='utf-8')
val_json = open(val_output_file, 'w', encoding='utf-8')
test_json.write('{\n')
val_json.write('{\n')

first_test = True
first_val = True

with open(input_file, 'rb') as f:
    for user_id, user_data in tqdm(ijson.kvitems(f, ''), desc="Writing OOD users"):
        # Check if user_id is in the sampled test or val set
        if user_id in ood_test_sampled:
            if not first_test:
                test_json.write(',\n')
            first_test = False
            json.dump({user_id: user_data}, test_json)
        elif user_id in ood_val_sampled:
            if not first_val:
                val_json.write(',\n')
            first_val = False
            json.dump({user_id: user_data}, val_json)

test_json.write('\n}\n')
val_json.write('\n}\n')
test_json.close()
val_json.close()

print("🎉 Out-of-distribution splits saved!")
print(f"✍️  {test_output_file}, {test_ids_file}")
print(f"✍️  {val_output_file}, {val_ids_file}")

# === CONFIGURATION FOR IN-DISTRIBUTION ===
print("🔍 Second pass: Collecting In-Distribution user IDs")
ind_val_output_file = "splits/ind_val_users.json"
ind_test_output_file = "splits/ind_test_users.json"
ind_val_ids_file = "splits/ind_val_user_ids.txt"
ind_test_ids_file = "splits/ind_test_user_ids.txt"
ind_val_user_limit = 1_000_000
ind_test_user_limit = 1_000_000
IN_DIST_TEST_YEAR_THRESHOLD = 2020
IN_DIST_VAL_YEAR_THRESHOLD = 2019

# === Pass 3: Identify in-distribution test/val user IDs which should have a training subset i.e. interactions in 2019 or earlier ===
ind_val_candidates = []
ind_test_candidates = []

print("🔍 Third pass: Scanning for In-Distribution test/val users")
seen_ids = set(ood_test_sampled).union(ood_val_sampled)

with open(input_file, 'rb') as f:
    for user_id, user_data in tqdm(ijson.kvitems(f, ''), desc="Scanning InD users"):
        if user_id in seen_ids:
            continue

        timestamps_lst = user_data.get("timestamps", [])
        if not timestamps_lst:
            continue

        timestamps_lst_reversed = list(reversed(timestamps_lst))

        for timestamp in timestamps_lst_reversed:
            timestamp = int(timestamp)
            first_timestamp = int(timestamps_lst[0])
            first_year = datetime.fromtimestamp(first_timestamp / 1000, tz=timezone.utc).strftime('%Y')
            year = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y')
            first_year = int(first_year)
            year = int(year)
            if year == IN_DIST_TEST_YEAR_THRESHOLD:
                if first_year < IN_DIST_VAL_YEAR_THRESHOLD:
                    ind_test_candidates.append(user_id)
                    ind_val_candidates.append(user_id)
                break
            elif year == IN_DIST_VAL_YEAR_THRESHOLD:
                if first_year < IN_DIST_VAL_YEAR_THRESHOLD:
                    ind_val_candidates.append(user_id)
                break

print(f"✅ Found {len(ind_test_candidates):,} InD test candidates")
print(f"✅ Found {len(ind_val_candidates):,} InD val candidates")

# === Random sampling with seed
ind_test_sampled = set(random.sample(ind_test_candidates, min(ind_test_user_limit, len(ind_test_candidates))))
# remove test users from val candidates
ind_val_candidates = [user_id for user_id in ind_val_candidates if user_id not in ind_test_sampled]
ind_val_sampled = set(random.sample(ind_val_candidates, min(ind_val_user_limit, len(ind_val_candidates))))
print(f"✅ Sampled {len(ind_test_sampled):,} InD test users")
print(f"✅ Sampled {len(ind_val_sampled):,} InD val users")

# === Save user IDs to text files ===
print("💾 Saving InD user IDs to text files")

# Save ID txt files
with open(ind_test_ids_file, 'w') as f_test_ids:
    f_test_ids.writelines(user_id + '\n' for user_id in ind_test_sampled)

with open(ind_val_ids_file, 'w') as f_val_ids:
    f_val_ids.writelines(user_id + '\n' for user_id in ind_val_sampled)

# === Final pass: Write JSON for InD users
print("💾 Final pass: Writing InD test/val users to JSON files")
ind_test_json = open(ind_test_output_file, 'w', encoding='utf-8')
ind_val_json = open(ind_val_output_file, 'w', encoding='utf-8')
ind_test_json.write('{\n')
ind_val_json.write('{\n')
first_ind_test = True
first_ind_val = True

with open(input_file, 'rb') as f:
    for user_id, user_data in tqdm(ijson.kvitems(f, ''), desc="Writing InD users"):
        if user_id in ind_test_sampled:
            if not first_ind_test:
                ind_test_json.write(',\n')
            first_ind_test = False
            json.dump({user_id: user_data}, ind_test_json)
        elif user_id in ind_val_sampled:
            if not first_ind_val:
                ind_val_json.write(',\n')
            first_ind_val = False
            json.dump({user_id: user_data}, ind_val_json)

ind_test_json.write('\n}\n')
ind_val_json.write('\n}\n')
ind_test_json.close()
ind_val_json.close()

print("🎉 In-distribution splits saved!")
print(f"✍️  {ind_test_output_file}, {ind_test_ids_file}")
print(f"✍️  {ind_val_output_file}, {ind_val_ids_file}")