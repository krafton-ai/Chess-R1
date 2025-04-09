#!/usr/bin/env bash

# Navigate to the folder containing your processed files
cd processed_dataset || exit 1

# Combine files from 0-14 into mate_train.jsonl
cat explain_dataset0{0..9}.jsonl explain_dataset1{0..4}.jsonl > mate_train.jsonl

# Use file 15 as mate_test.jsonl
cp explain_dataset15.jsonl mate_test.jsonl

echo "mate_train.jsonl and mate_test.jsonl have been created."