import sqlite3
import os

def load_metadata(sqlite_db_path):
    """Return dict {asin: title} for all non-empty titles in your metadata table."""
    conn = sqlite3.connect(sqlite_db_path)
    cur = conn.cursor()
    cur.execute("SELECT asin, title FROM metadata")
    rows = cur.fetchall()
    conn.close()
    return {asin: title for asin, title in rows if title}

def write_item_file(asins, output_path):
    """
    Write a RecBole .item atomic file listing each ASIN.
    asins: iterable of ASIN strings
    output_path: e.g. "/path/to/recbole_dataset_sampled.item"
    """
    with open(output_path, "w") as out:
        # Header: feature_name:feat_type
        out.write("item_id:token\n")  # single discrete feature :contentReference[oaicite:4]{index=4}
        for asin in sorted(asins):
            out.write(f"{asin}\n")

if __name__ == "__main__":
    # 1) Load metadata ASINs from your SQLite DB
    sqlite_path = "amazon_parquet_data/metadata_titles.db"
    meta = load_metadata(sqlite_path)
    
    # 2) Optionally intersect with the ASINs you actually sample in train/val/test
    #    to avoid useless entries—but not strictly necessary.
    #    For example, if you have lists of sampled ASINs:
    # sampled_asins = set(load_txt("ind_train_user_ids_sampled.txt")) ∪ … 
    # all_needed_asins = sampled_asins  # or union of all splits
    all_needed_asins = set(meta.keys())
    
    # 3) Write the .item file into your data folder
    output_item_file = os.path.join("recbole_dataset_sampled", "recbole_dataset_sampled.item")
    os.makedirs(os.path.dirname(output_item_file), exist_ok=True)
    write_item_file(all_needed_asins, output_item_file)
    print(f"Wrote {len(all_needed_asins)} items to {output_item_file}")
