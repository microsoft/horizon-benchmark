#!/bin/bash

# Ensure script stops on error
set -e

# Define Python script name
PYTHON_SCRIPT="data/merging_user_history_phase_one.py"  # Change this to your actual script filename

# Define output log file
LOG_FILE="merging_process_phase_one.log"

# Run first python script
echo "🚀 Running first script..."
python3 "$PYTHON_SCRIPT" --start 0 --end 7 | tee -a "$LOG_FILE"
echo "✅ Completed first script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

echo "🚀 Running second script..."
python3 "$PYTHON_SCRIPT" --start 7 --end 10 | tee -a "$LOG_FILE"
echo "✅ Completed second script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

echo "🚀 Running third script..."
python3 "$PYTHON_SCRIPT" --start 10 --end 16 | tee -a "$LOG_FILE"
echo "✅ Completed third script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

echo "🚀 Running fourth script..."
python3 "$PYTHON_SCRIPT" --start 16 --end 24 | tee -a "$LOG_FILE"
echo "✅ Completed fourth script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

# Run second python script
echo "🚀 Running fifth script..."
python3 "$PYTHON_SCRIPT" --start 24 --end 32 | tee -a "$LOG_FILE"
echo "✅ Completed fifth script"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"