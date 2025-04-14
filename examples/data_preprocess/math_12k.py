#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
and filter out samples where the question length exceeds 512 characters.
"""

import os
import datasets
import argparse

# Template functions for question formatting
def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: " + question +
        "\nAssistant: <think>"
    )

def apply_no_template(question: str):
    return question

# Dummy implementation of extract_solution (replace with your real logic if needed)
def extract_solution(answer_raw: str):
    return answer_raw.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/jovyan/chessLLM/data/understand_r1_raw/math_12k")
    parser.add_argument("--save_dir", default="/home/jovyan/chessLLM/data/understand_r1_math_12k/")
    parser.add_argument("--template_type", default='qwen_math_template')
    args = parser.parse_args()

    # Select the appropriate template function based on template_type argument
    if args.template_type == 'qwen_math_template':
        template_func = apply_qwen_math_template
    elif args.template_type == 'r1_template':
        template_func = apply_r1_template
    elif args.template_type == 'no_template':
        template_func = apply_no_template
    else:
        raise ValueError("Unsupported template type: {}".format(args.template_type))

    # Load the dataset
    dataset = datasets.load_from_disk(args.local_dir)
    train_dataset = dataset["train"]

    # Filter out samples where the question length (the "problem" field) exceeds 1024 characters.
    train_dataset = train_dataset.filter(lambda x: len(x["problem"]) <= 1024)
    print("Number of samples after filtering (<=1024 characters):", train_dataset.num_rows)

    # Process each example: apply the selected template to the question.
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")
            
            # Apply the selected template function to the question.
            question = template_func(question_raw)
            answer_raw = example.pop("answer")

            data = {
                "data_source": "math_12k",
                "template_type": args.template_type,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math_12k",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": example.pop("solution"),
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    # Transform the dataset with indices.
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    # Save the processed dataset as a Parquet file.
    output_path = os.path.join(args.save_dir, args.template_type, "train.parquet")
    train_dataset.to_parquet(output_path)
    print('Preprocessing complete!')
