import ijson
import json
from tqdm import tqdm

input_file = 'amazon_parquet_data/merged_users_all_final_filtered.json'
output_file = 'amazon_parquet_data/merged_users_all_final_filtered_no_reviews.json'

def strip_reviews(input_path, output_path):
    with open(input_path, 'rb') as in_f, open(output_path, 'w', encoding='utf-8') as out_f:
        parser = ijson.kvitems(in_f, '')

        out_f.write("{\n")
        first = True
        count = 0

        for user_id, user_data in tqdm(parser, desc="Processing users (removing reviews)"):
            count += 1

            # Remove 'reviews' key if present
            user_data.pop('reviews', None)

            # Write comma if not first
            if not first:
                out_f.write(",\n")
            first = False

            # Write the user entry
            record = json.dumps({user_id: user_data}, ensure_ascii=False)
            out_f.write(record[1:-1])  # remove outermost braces

        out_f.write("\n}\n")

    print(f"\n✅ Finished writing {count} users to {output_path}")

if __name__ == '__main__':
    strip_reviews(input_file, output_file)
