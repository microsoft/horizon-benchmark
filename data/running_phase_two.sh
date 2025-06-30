#!/bin/bash

# Ensure script stops on error
set -e

# Define Python script name
PYTHON_SCRIPT="data/merging_user_history_phase_two_large.py"  # Change this to your actual script filename

# Define output log file
LOG_FILE="merging_process_phase_two.log"

# Run first python script
echo "🚀 Running first script..."
python3 "$PYTHON_SCRIPT" --input_file_one "merged_users_[0-7].parquet" --input_file_two "merged_users_[7-10].parquet" --output_file "merged_users_[0-10].parquet" | tee -a "$LOG_FILE"
echo "✅ Completed first script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

echo "🚀 Running second script..."
python3 "$PYTHON_SCRIPT" --input_file_one "merged_users_[0-10].parquet" --input_file_two "merged_users_[10-16].parquet" --output_file "merged_users_[0-16].parquet" | tee -a "$LOG_FILE"
echo "✅ Completed second script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

# Run third python script
echo "🚀 Running third script..."
python3 "$PYTHON_SCRIPT" --input_file_one "merged_users_[16-24].parquet" --input_file_two "merged_users_[24-32].parquet" --output_file "merged_users_[16-32].parquet" | tee -a "$LOG_FILE"
echo "✅ Completed third script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

# Run fourth python script
echo "🚀 Running fourth script..."
python3 "$PYTHON_SCRIPT" --input_file_one "merged_users_test_[0-16].parquet" --input_file_two "merged_users_test_[16-32].parquet" --output_file "merged_users_all_final.parquet" | tee -a "$LOG_FILE"
echo "✅ Completed fourth script"
echo "--------------------------------------------"
echo "--------------------------------------------" 
echo "--------------------------------------------"

# Convert final parquet file to JSON
echo "🚀 Converting final parquet file to JSON..."
python3 data/parquet_to_json.py