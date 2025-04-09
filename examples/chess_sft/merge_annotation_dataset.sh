#!/usr/bin/env bash

# Navigate to the folder containing your processed files
cd processed_dataset || exit 1

# Combine files from 0-14 into mate_train.jsonl
cat annotated_pgn-data.jsonl-0000{0..1}-of-00002 > annotation_train.jsonl

echo "annotation_train.jsonl have been created."