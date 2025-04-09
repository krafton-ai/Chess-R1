#!/usr/bin/env bash

# Navigate to the folder containing your processed files
cd processed_dataset || exit 1

# Create (or ensure the existence of) a parent-level merged_dataset folder
mkdir -p ../merged_dataset

# Merge all annotation files into the merged_dataset folder
cat annotated_pgn-data.jsonl-0000{0..1}-of-00002 > ../merged_dataset/chess_annotation_train.json
echo "chess_annotation_train.json created in ../merged_dataset."

# Merge best move files
cat chess_best_move_0{0..5}.json > ../merged_dataset/chess_best_move_train.json
echo "chess_best_move_train.json created in ../merged_dataset."

# Merge comparison files
cat chess_comparison_0{6..9}.json chess_comparison_1{0..4}.json > ../merged_dataset/chess_comparison_train.json
echo "chess_comparison_train.json created in ../merged_dataset."

# Copy comparison_dataset_15.json into merged_dataset (as test set)
cp chess_comparison_15.json ../merged_dataset/chess_comparison_test.json
echo "chess_comparison_test.json copied to ../merged_dataset."