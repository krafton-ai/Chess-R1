#!/usr/bin/env bash

# transform_all.sh
# This script iterates over explain_dataset0.jsonl to explain_dataset15.jsonl
# and calls the transform_dataset script for each file.

for i in $(seq -f "%02g" 6 15); do
    INPUT_FILE="./raw_dataset/explain_dataset${i}_both.jsonl"
    OUTPUT_FILE="./processed_dataset/chess_comparison_${i}.json"
    
    echo "Processing $INPUT_FILE => $OUTPUT_FILE"
    python generate_comparison_dataset.py --input_file="$INPUT_FILE" --output_file="$OUTPUT_FILE"
done