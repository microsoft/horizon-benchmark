#!/bin/bash

# Ensure script stops on error
set -e

# Run first script to curate metadata from source
echo "🚀 Running first script to curate metadata from source..."
python3 data/metadata_curation.py | tee -a "metadata_curation.log"
echo "✅ Completed first script to curate metadata from source"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

# Run second script to create sqlite database from json file for faster processing
echo "🚀 Running second script to create sqlite database from json file for faster processing..."
python3 data/metadata_json_to_sqlite.py | tee -a "create_sqlite_db.log"
echo "✅ Completed second script to create sqlite database from json file for faster processing"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

# Running third script to filter users from main data where the users have missing metadata
echo "🚀 Running third script to filter users from main data where the users have missing metadata..."
python3 data/filtering_json.py | tee -a "filter_users_missing_metadata.log"
echo "✅ Completed third script to filter users from main data where the users have missing metadata"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"

# Running fourth script to remove reviews to create a lighter version of the dataset
echo "🚀 Running fourth script to remove reviews to create a lighter version of the dataset..."
python3 data/remove_reviews.py | tee -a "remove_reviews.log"
echo "✅ Completed fourth script to remove reviews to create a lighter version of the dataset"
echo "--------------------------------------------"
echo "--------------------------------------------"
echo "--------------------------------------------"