import sqlite3
import json
import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import List
import numpy as np
import math
import gc
import time
from accelerate import Accelerator
import argparse

# --------------------- SETUP ---------------------
torch.backends.cudnn.benchmark = True
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
rank = accelerator.process_index
world_size = accelerator.num_processes
print(f"✅ Process {rank} running on {device} | world size: {world_size}")

# --------------------- PATHS ---------------------
OUTPUT_DIR = "evaluation/shards_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CATALOG_TITLES_PATH = "cache/catalog_titles.txt"
CATALOG_EMBEDDINGS_PATH = "cache/catalog_embeddings.npy"

# --------------------- SETTINGS ---------------------
BATCH_SIZE = 4096

# --------------------- LOAD MODEL ---------------------
if accelerator.is_main_process:
    print("🔄 Loading model in mixed precision fp16...")
with accelerator.main_process_first():
    tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
    model = AutoModel.from_pretrained("hyp1231/blair-roberta-base", trust_remote_code=True)
model = model.to(device)
model.eval()
model = accelerator.prepare(model)

# --------------------- FUNCTIONS ---------------------
def fetch_titles_from_sqlite(db_path: str) -> List[str]:
    if accelerator.is_main_process:
        print(f"🔍 Fetching titles from {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT title FROM metadata")
    rows = cur.fetchall()
    conn.close()
    titles = [row[0] for row in rows]
    if accelerator.is_main_process:
        print(f"✅ Retrieved {len(titles)} titles from DB")
    return titles


def deduplicate_and_clean(titles: List[str]) -> List[str]:
    seen = set()
    unique = []
    for t in tqdm(titles, desc="🧹 Cleaning & deduplicating", disable=not accelerator.is_main_process):
        t_clean = t.replace("\n", " ").replace("\r", " ").strip()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            unique.append(t_clean)
    if accelerator.is_main_process:
        print(f"✅ {len(unique)} unique titles after cleaning")
    return unique


def embed_and_save_shard(titles: List[str], prefix: str):
    total = len(titles)
    per_proc = math.ceil(total / world_size)
    start = rank * per_proc
    end = min(start + per_proc, total)
    local_titles = titles[start:end]
    if accelerator.is_main_process:
        print(f"🔁 {prefix}: splitting {total} → {world_size} shards (~{per_proc}/shard)")
    print(f"[rank {rank}] Embedding {prefix} titles {start}:{end} (~{len(local_titles)})")

    local_embs = []
    for i in tqdm(range(0, len(local_titles), BATCH_SIZE), desc=f"{prefix} shard {rank} embedding", disable=not accelerator.is_local_main_process):
        batch = local_titles[i: i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0]
            emb = emb / emb.norm(dim=1, keepdim=True)
        local_embs.append(emb.detach().cpu().numpy())
        del inputs, outputs, emb
        gc.collect()
        torch.cuda.empty_cache()

    local_emb_np = np.vstack(local_embs) if local_embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_embeddings_shard_{rank}.npy"), local_emb_np)
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_titles_shard_{rank}.txt"), 'w') as f:
        for t in tqdm(local_titles, desc=f"Writing {prefix} titles shard {rank}", disable=not accelerator.is_local_main_process):
            t_clean = t.replace("\n", " ").replace("\r", " ").strip()
            f.write(t_clean + '\n')

    if accelerator.is_main_process:
        print(f"✅ Saved {prefix} shard {rank}: {len(local_titles)} titles, {local_emb_np.shape[0]} embeddings")


def merge_shards(prefix: str, num_shards: int, final_titles_path: str, final_embed_path: str):
    if not accelerator.is_main_process:
        return
    print(f"🔄 Waiting for {num_shards} {prefix} shard files...")
    while True:
        missing = []
        for r in range(num_shards):
            tfile = os.path.join(OUTPUT_DIR, f"{prefix}_titles_shard_{r}.txt")
            efile = os.path.join(OUTPUT_DIR, f"{prefix}_embeddings_shard_{r}.npy")
            if not os.path.exists(tfile) or not os.path.exists(efile):
                missing.append(r)
        if not missing:
            break
        print(f"  Still waiting on shards: {missing[:5]}{'...' if len(missing)>5 else ''}")
        time.sleep(10)

    print(f"🔄 Merging {prefix} shards...")
    all_titles = []
    all_embs = []
    for r in tqdm(range(num_shards), desc=f"Merging {prefix} shards"):  
        with open(os.path.join(OUTPUT_DIR, f"{prefix}_titles_shard_{r}.txt"), 'r') as f:
            all_titles.extend([l.strip() for l in f])
        all_embs.append(np.load(os.path.join(OUTPUT_DIR, f"{prefix}_embeddings_shard_{r}.npy")))
    all_embs_np = np.vstack(all_embs)

    assert len(all_titles) == all_embs_np.shape[0], f"Mismatch after merge: {len(all_titles)} vs {all_embs_np.shape[0]}"
    np.save(final_embed_path, all_embs_np)
    with open(final_titles_path, 'w') as f:
        for t in tqdm(all_titles, desc=f"Writing merged {prefix} titles"):  
            t_clean = t.replace("\n", " ").replace("\r", " ").strip()
            f.write(t_clean + '\n')
    print(f"🎉 Merged {prefix}: {len(all_titles)} titles, {all_embs_np.shape[0]} embeddings saved")

"""
Function to keep GPUs warm
"""
def KEEP_WARM():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time
    print("🔥 Loading GPT-2 on 1 GPU (A100)...")
    model_name = "gpt2-xl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()

    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    print("🚀 GPT-2 loaded. Starting infinite generation loop to keep GPU warm...")

    try:
        while True:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=64,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50
                )
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                #print(f"> {text}\n")
            time.sleep(3)
    except KeyboardInterrupt:
        print("🛑 Exiting GPU warm-up loop.")


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings for catalog and predicted titles")
    parser.add_argument("--db_path", type=str, default='sqlite.db', help="Path to the SQLite database")
    parser.add_argument("--predictions_jsonl", type=str, default='preds.jsonl', help="Path to the predictions JSONL file")
    parser.add_argument("--embeddings_path", type=str, default='embeddings.npy', help="Path to save the embeddings")
    parser.add_argument("--titles_path", type=str, default='titles.txt', help="Path to save the titles")
    args = parser.parse_args()

    print(f"Arguments: {args}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Catalog titles
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"Database file not found: {args.db_path}")
    db_path = args.db_path
    predictions_jsonl = args.predictions_jsonl
    predicted_embeddings_file_path = args.embeddings_path
    predicted_titles_file_path = args.titles_path

    # Catalog
    raw_titles = fetch_titles_from_sqlite(db_path)
    titles = deduplicate_and_clean(raw_titles)
    if not (os.path.exists(CATALOG_TITLES_PATH) and os.path.exists(CATALOG_EMBEDDINGS_PATH)):
        embed_and_save_shard(titles, "catalog")
        merge_shards("catalog", world_size, CATALOG_TITLES_PATH, CATALOG_EMBEDDINGS_PATH)
    else:
        print(f"✅ Catalog titles and embeddings already exist at {CATALOG_TITLES_PATH} and {CATALOG_EMBEDDINGS_PATH}")

    # Predicted titles
    if not os.path.exists(predictions_jsonl):
        raise FileNotFoundError(f"Predictions JSONL file not found: {predictions_jsonl}")

    # Predicted: now list of lists
    raw_pred = []
    with open(predictions_jsonl, 'r') as f:
        for line in tqdm(f, desc="Reading predicted JSONL", disable=not accelerator.is_main_process):
            data = json.loads(line)
            for group in data.get("predicted_history_titles", []):  # group is a list of strings
                raw_pred.extend(group)
    print(f"✅ Read {len(raw_pred)} predicted titles from JSONL")
    pred_titles = deduplicate_and_clean(raw_pred)
    print(f"✅ {len(pred_titles)} unique predicted titles after cleaning")
    embed_and_save_shard(pred_titles, "predicted")
    merge_shards("predicted", world_size, predicted_titles_file_path, predicted_embeddings_file_path)

    # Cleanup shards
    if accelerator.is_main_process:
        print("🧹 Cleaning up shard files...")
        for r in range(world_size):
            for prefix in ["catalog", "predicted"]:
                if os.path.exists(os.path.join(OUTPUT_DIR, f"{prefix}_titles_shard_{r}.txt")):
                    print(f"🗑️ Deleting {prefix} shard {r} files...")
                    os.remove(os.path.join(OUTPUT_DIR, f"{prefix}_titles_shard_{r}.txt"))
                else:
                    print(f"✅ {prefix} shard {r} files not found, skipping deletion.")
                if os.path.exists(os.path.join(OUTPUT_DIR, f"{prefix}_embeddings_shard_{r}.npy")):
                    print(f"🗑️ Deleting {prefix} shard {r} files...")
                    os.remove(os.path.join(OUTPUT_DIR, f"{prefix}_embeddings_shard_{r}.npy"))
                else:
                    print(f"✅ {prefix} shard {r} files not found, skipping deletion.")
        print("✅ All done!")

if __name__ == "__main__":
    try:
        main()
        print(f"✅ All done!")
        KEEP_WARM()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        KEEP_WARM()
        print("❌ An error occurred during the execution of the script. Please check the logs for more details.")
