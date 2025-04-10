""" Preprocess dataset for chess task """

import os
import argparse
import json
import re
import ipdb

from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs

def jsonl_to_list(jsonl_file_path):
    """
    Reads a JSONL file and returns a list of JSON objects.
    
    Each line in the file is expected to contain a valid JSON object.
    
    :param jsonl_file_path: Path to the JSONL file.
    :return: A list of JSON objects.
    """
    result = []
    
    with open(jsonl_file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}\nError: {e}")
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="./data/")
    parser.add_argument('--data_path', default='/home/jovyan/chessLLM/verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data/')
    parser.add_argument('--jsonl', action='store_true')
    args = parser.parse_args()

    if args.jsonl:
        data_types = ["chess_comparison", "chess_best_move",]
    else:
        data_types = ["chess_modeling_instruct"]
    
    for data_type in data_types:
        # Modify save_dir to include template type
        save_dir = args.save_dir + data_type
        
        if args.jsonl:
            train_data_path = f"{args.data_path}{data_type}_train.jsonl"
            test_data_path = f"{args.data_path}{data_type}_test.jsonl"

            train_data = jsonl_to_list(train_data_path)
            train_dataset = Dataset.from_list(train_data)

            test_dataset = None
            if os.path.exists(test_data_path):
                with open(test_data_path, 'rb') as f:
                    test_data = jsonl_to_list(test_data_path)
                test_dataset = Dataset.from_list(test_data)
            else:
                print("Test dataset not found. Skipping test processing.")
        else:
            train_data_path = f"{args.data_path}{data_type}_train.json"
            test_data_path = f"{args.data_path}{data_type}_test.json"
            with open(train_data_path, 'rb') as f:
                train_data = json.load(f)

            train_dataset = Dataset.from_list(train_data)
            # Check if the test file exists before loading.
            test_dataset = None
            if os.path.exists(test_data_path):
                with open(test_data_path, 'rb') as f:
                    test_data = json.load(f)
                test_dataset = Dataset.from_list(test_data)
            else:
                print("Test dataset not found. Skipping test processing.")

        def convert_fn(example):
            # Retrieve the keys, providing defaults if needed.
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            system_text = example.get("system", "")
            
            prompt = f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

            answer_pattern = r'<answer>(.*?)</answer>'
            matches = list(re.finditer(answer_pattern, output_text, re.DOTALL))
            final_answer = matches[-1].group(1).strip()

            return prompt, final_answer

        def make_map_fn(split):
            def process_fn(example, idx):
                question, answer = convert_fn(example)
                solution = {
                    "answer": answer,
                }
                data = {
                    "data_source": data_type,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "chess",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data
            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        # Process the test dataset only if it was successfully loaded.
        if test_dataset is not None:
            test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        # Create local directory if not exists
        os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

        train_filename = "train.parquet"
        test_filename = "test.parquet"

        # Save the processed training dataset.
        train_dataset.to_parquet(os.path.join(save_dir, train_filename))
        
        # Save the processed test dataset only if available.
        if test_dataset is not None:
            test_dataset.to_parquet(os.path.join(save_dir, test_filename))
