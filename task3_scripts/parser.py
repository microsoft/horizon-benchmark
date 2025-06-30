import json
import re
import logging
import argparse
import os
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='parser.log',
                    filemode='w')
# Create a logger
logger = logging.getLogger(__name__)

FAILED_USER_IDS = set()

def extract_queries_from_string(s):
    """
    Extracts item_descriptions_timewise from a possibly malformed JSON string.
    Returns all complete, properly quoted query strings found under "item_descriptions_timewise".
    """
    # Step 1: Truncate after first triple-backtick if present
    s = s.split("```")[0].strip()

    # Step 2: Try normal JSON parsing
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "item_descriptions_timewise" in obj:
            item_descriptions_timewise = obj["item_descriptions_timewise"]
            if isinstance(item_descriptions_timewise, list):
                return [q for q in item_descriptions_timewise if isinstance(q, str)]
    except Exception:
        pass

    # Step 3: Fallback: regex extract text after "item_descriptions_timewise": [
    item_descriptions_timewise_section = re.search(r'"item_descriptions_timewise"\s*:\s*\[(.*?)\]', s, re.DOTALL)
    if item_descriptions_timewise_section:
        raw_content = item_descriptions_timewise_section.group(1)
        # Extract only complete quoted strings (not ending mid-way)
        return re.findall(r'"(.*?)"', raw_content)

    # Step 4: Fallback: if above failed, try to find all quoted strings globally after "item_descriptions_timewise":
    item_descriptions_timewise_index = s.find('"item_descriptions_timewise"')
    if item_descriptions_timewise_index != -1:
        after_item_descriptions_timewise = s[item_descriptions_timewise_index:]
        return re.findall(r'"(.*?)"', after_item_descriptions_timewise)

    return []  # Nothing valid found

def process_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            user_id = data.get("user_id")
            parsed_predictions = []

            for pred_str in data.get("predicted_history_titles", []):
                queries = extract_queries_from_string(pred_str)
                if len(queries) == 0:
                    logger.warning(f"No queries found in prediction string: {pred_str}")
                    logger.info(f"User ID: {user_id}")
                    logger.info(f"Original string: {pred_str}")
                    FAILED_USER_IDS.add(user_id)
                else:
                    # Remove duplicates while preserving order
                    parsed_predictions.append(queries)

            output_data = {
                "user_id": data["user_id"],
                "true_history_titles": data["true_history_titles"],
                "true_history_asins": data["true_history_asins"],
                "predicted_history_titles": parsed_predictions
            }

            outfile.write(json.dumps(output_data) + "\n")

# Example usage:
# process_jsonl("input_file.jsonl", "parsed_output.jsonl")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process JSONL files to extract queries.")
    parser.add_argument("--input_file", type=str, help="Input JSONL file path")
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.isfile(args.input_file):
        logger.error(f"Input file {args.input_file} does not exist.")
        print(f"Input file {args.input_file} does not exist.")
        exit(1)

    # Output file by renaming the input file
    input_file_jsonl = args.input_file
    output_file_jsonl = input_file_jsonl.replace(".jsonl", "_parsed.jsonl")
    process_jsonl(input_file_jsonl, output_file_jsonl)
    logger.info(f"Processing complete. Failed user IDs: {FAILED_USER_IDS}")
    print(f"Processing complete.")
    print(f'Number of failed user IDs: {len(FAILED_USER_IDS)}')
    print(f'Failed user IDs: {FAILED_USER_IDS}')
    print(f"Output written to {output_file_jsonl}")
    print(f"Log file created: parser.log")