import argparse
import json
import sqlite3
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import yaml
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# -------------------- SETUP --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="gemma2_9b_long_horizon.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

TEST_THRESHOLD = '1577836800'  # Jan 1, 2020
BATCH_SIZE = 512

# -------------------- HELPERS --------------------
def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_title(cursor, asin):
    try:
        cursor.execute("SELECT title FROM metadata WHERE asin = ?", (asin,))
        res = cursor.fetchone()
        return res[0] if res else None
    except:
        return None

def get_titles_bulk(cursor, asins):
    q_marks = ",".join(["?"] * len(asins))

    try:
        cursor.execute(f"SELECT asin, title FROM metadata WHERE asin IN ({q_marks})", tuple(asins))
        return dict(cursor.fetchall())
    except Exception as e:
        print(f"Error fetching titles: {e}")
        return {}

def build_prompt(history_titles, item_sep="|<SEP>|"):
    """
    Build a concise zero‑shot prompt for LLaMA‑3.1‑8B that:
      1. Asks for a one‑sentence guideline
      2. Asks for exactly 10 personalized next‑item descriptions in JSON

    Args:
      history_titles (List[str]): sequence of past product titles
      item_sep (str): separator token between titles
      
    Returns:
      str: the full prompt to feed to the model
    """
    joined = f" {item_sep} ".join(history_titles)
    return (
    f"<im_start>system\nYou are an expert at predicting the next products a user may want based on their Amazon product history.\n<|im_end|>\n"
    f"<im_start>user\nHistory: {joined} {item_sep} This was the user's Amazon product history with exact product titles (NOT descriptions).\n"
    f"Your task is to generate descriptions for the next 10 items the user is most likely to be interested in. Provide concise, one‑sentence descriptions that capture the essence of each potential item. These will guide recommendation generation.\n"
    f"Try to model the sequences in the user history and provide a mix of relevant and serendipitous items trying to capture the user's interests, intents and changes in behaviour. Use the first item description to guide your next timestep's item description generation in an autoregressive manner.\n"
    f"1. Write a one-sentence guideline explaining the patterns or preferences you observed in the user history that informed your item descriptions.\n"
    f"2. Provide exactly 10 next-item descriptions balancing relevance and serendipity generated one after the other in temporal order.\n\n"
    f"## Output Format\nProvide the response only as a JSON object with two fields: (do not generate anything else)\n"
    f"```json\n"
    f"{{\n"
    f"  \"guideline\": \"a one-sentence guideline summarizing the user's preferences and your strategy for the temporal item selection for the next 10 timesteps.\",\n"
    f"  \"item_descriptions_timewise\": [\n"
    f"    \"item_description_time_step1\",\n"
    f"    \"item_description_time_step2\",\n"
    f"    ...,\n"
    f"    \"item_description_time_step10\"\n"
    f"  ]\n"
    f"}}\n"
    f"```\n"
    f"<|im_end|>\n"
    f"<|im_start|>assistant\n[assistant](#next_item_descriptions)\n```json"
)
# -------------------- MAIN --------------------
def run_prediction(input_json, sqlite_db_path, output_path, is_ood, MODEL_NAME, NUM_GPUs, config_object):
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("🔍 Tokenizer loaded.")

    conn = sqlite3.connect(sqlite_db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    cur = conn.cursor()

    user_data = load_json_file(input_json)
    print(f"🔍 Loaded {len(user_data)} users from {input_json}")

    #Take the only first 10 right now
    # user_ids_sampled = list(user_data.keys())[:30]
    # user_data = {user_id: user_data[user_id] for user_id in user_ids_sampled}
    # print(f"🔍 Sampled {len(user_data)} users for testing.")
    
    # Track user processing stats
    total_users = len(user_data)
    users_skipped_titles = 0
    users_skipped_threshold = 0
    users_processed = 0
    users_with_llm_errors = set()
    
    # Keep track of which users have been written to the output file
    processed_users = set()

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=NUM_GPUs  # multi-GPU if available
    )

    # Constants
    max_model_len = 8192  # Model's context length
    max_tokens = config_object["model"].get("max_tokens", 220)  # Max tokens to generate

    sampling_params = SamplingParams(
        temperature=config_object["model"].get("temperature", 0.7),
        top_p=config_object["model"].get("top_p", 0.95),
        max_tokens=config_object["model"].get("max_tokens", 220),
    )

    print("Initialized LLM.")
    print(f"Model: {MODEL_NAME}")
    print(f"Number of GPUs: {NUM_GPUs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output path: {output_path}")
    print(f"SQLite DB path: {sqlite_db_path}")
    print(f"Input JSON path: {input_json}")
    print(f"Is OOD: {is_ood}")

    # Process users in batches to collect prompts
    prompt_buffer = []
    meta_buffer = []
    
    # Store all results by user_id for final write
    all_results = {}

    def flush_batch(outfile):
        nonlocal prompt_buffer, meta_buffer, users_with_llm_errors
        
        if not prompt_buffer:
            return
            
        try:
            outputs = llm.generate(prompt_buffer, sampling_params)
            
            for i, output in enumerate(outputs):
                meta = meta_buffer[i]
                user_id = meta["user_id"]
                target_titles = meta["target_titles"]
                target_asins = meta["target_asins"]

                raw = output.outputs[0].text.strip()
                prediction = raw
                
                if user_id not in all_results:
                    all_results[user_id] = {
                        "user_id": user_id,
                        "true_history_titles": [],
                        "true_history_asins": [],
                        "predicted_history_titles": []
                    }

                all_results[user_id]["true_history_titles"].extend(target_titles)
                all_results[user_id]["true_history_asins"].extend(target_asins)
                all_results[user_id]["predicted_history_titles"].append(prediction)

        except Exception as e:
            print(f"❌ LLM error in batch: {e}")
            logger.error(f"LLM error in batch: {e}")
            # Mark all users in this batch as having errors
            for meta in meta_buffer:
                users_with_llm_errors.add(meta["user_id"])
        finally:
            prompt_buffer.clear()
            meta_buffer.clear()

    # First pass: process all users and collect prompts
    for user_id, user_info in tqdm(user_data.items(), desc="Processing users"):
        asin_hist = user_info.get("history", [])
        timestamps = user_info.get("timestamps", [])

        if len(asin_hist) < 2:
            users_skipped_titles += 1
            logger.info(f"User {user_id} skipped: history too short")
            continue

        # Get all titles for the user's history
        title_history = [get_title(cur, asin) for asin in asin_hist]
        if any(t is None for t in title_history):
            users_skipped_titles += 1
            logger.info(f"User {user_id} skipped: missing titles")
            continue

        # Check if any items are after the threshold date
        start_idx = next((i for i, ts in enumerate(timestamps) if ts >= TEST_THRESHOLD), len(timestamps))
        if start_idx == len(timestamps):
            users_skipped_threshold += 1
            logger.info(f"User {user_id} skipped: insufficient history before threshold")
            continue
        elif start_idx == 0:
            start_idx = 1  # Ensure at least one item is before the threshold
            
        # User is valid for processing
        users_processed += 1

        # Generate prompts for the one step of long horizon prediction
        if len(title_history) - start_idx > 10:
            target_titles = title_history[start_idx:start_idx + 10]
            target_asins = asin_hist[start_idx:start_idx + 10]
        else:
            target_titles = title_history[start_idx:]
            target_asins = asin_hist[start_idx:]
        history_titles = title_history[:start_idx]
        prompt = build_prompt(history_titles)

        # Check if the prompt is too long and truncate if necessary
        prompt_ids = tokenizer(prompt)["input_ids"]
        if len(prompt_ids) + max_tokens > max_model_len:
            tokens_to_keep = max_model_len - max_tokens
            prompt_ids = prompt_ids[-tokens_to_keep:]  # Left-truncate
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # Add the prompt and metadata to the buffers
        prompt_buffer.append(prompt)
        meta_buffer.append({
            "user_id": user_id,
            "target_titles": target_titles,
            "target_asins": target_asins
        })

        # Flush when batch is full
        if len(prompt_buffer) >= BATCH_SIZE:
            flush_batch(outfile=None)  # No writing to file yet

    # Final flush for remaining prompts
    if prompt_buffer:
        flush_batch(outfile=None)
    
    # Now write all results to file
    with open(output_path, 'w') as outfile:
        for user_id, result in all_results.items():
            if user_id not in users_with_llm_errors:
                json.dump(result, outfile)
                outfile.write("\n")
                processed_users.add(user_id)

    print(f"\n✅ Output saved to: {output_path}")
    print(f"📊 Stats:")
    print(f"Total users in input: {total_users}")
    print(f"Users skipped due to missing/short titles: {users_skipped_titles}")
    print(f"Users skipped due to insufficient history before threshold: {users_skipped_threshold}")
    print(f"Users eligible for processing: {users_processed}")
    print(f"Users with LLM errors: {len(users_with_llm_errors)}")
    print(f"Users successfully written to output: {len(processed_users)}")
    print(f"Total user entries in output file: {len(processed_users)}")

    conn.close()


"""
Function to keep GPUs warm
"""
def KEEP_WARM():
    print("🔥 Loading GPT-2 on 1 GPU (A100)...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

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

# -------------------- ENTRY --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        run_prediction(
            input_json=config["input_json"],
            sqlite_db_path=config["sqlite_db"],
            output_path=config["output_path"],
            is_ood=config.get("is_ood", False),
            MODEL_NAME=config["model"]["name"],
            NUM_GPUs=config["model"].get("num_gpus", 1),
            config_object=config
        )
        print("\n✅ Finished processing all users.")
        print("💾 Output saved to:", config["output_path"])
        KEEP_WARM()

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error: {e}")
        print("An error occurred while processing the data. Please check the logs for more details.")
        KEEP_WARM()
        print("🔥 Keeping GPU warm...")