""" Preprocess dataset for chess task """

import os
import argparse
import json
import ipdb

from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.chess import MOVE_TO_ACTION_DICT

from searchless_chess.src import bagz
from searchless_chess.src import constants

import chess

def make_prefix(fen, template_type, include_legal_moves=True):
    board = chess.Board(fen)
    legal_moves = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION_DICT[x.uci()])
    legal_moves_list = " ".join(move.uci() for move in legal_moves)

    legal_moves_text = f"The available legal moves are: {legal_moves_list}" if include_legal_moves else ""

    if template_type == 'base_with_legal_move':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The User asks the best move to make for a given chess board state, and the Assistant solves it. The Assistant is a professional chess player who first thinks about the reasoning process in the mind and then provides the User with the answer. The Assistant’s reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. The reasoning process should describe how the Assistant analyzes the position and decide on the best move. The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included. Now, the user provides a FEN string and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
User: The current FEN string is {fen}. {legal_moves_text}. 
Assistant: \nLet me solve this step by step.\n<think>
"""
    elif template_type == 'base_wo_legal_move':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The User asks the best move to make for a given chess board state, and the Assistant solves it. The Assistant is a professional chess player who first thinks about the reasoning process in the mind and then provides the User with the answer. The Assistant’s reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. The reasoning process should describe how the Assistant analyzes the position and decide on the best move. The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included. Now, the user provides a FEN string.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
User: The current FEN string is {fen}. 
Assistant: \nLet me solve this step by step.\n<think>
"""
    elif template_type == 'llama_instruct_with_legal_move':
        prefix = f"""\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included.
Now, the user provides a FEN string and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
<|eot_id|><|start_header_id|>user<|end_header_id|>
Current FEN string: {fen}
{legal_moves_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|><|begin_of_text|>\nLet me solve this step by step.\n<think>
"""
    elif template_type == 'qwen_instruct_with_legal_move':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included.
Now, the user provides a FEN string and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
<|im_end|>
<|im_start|>user
Current FEN string: {fen}
{legal_moves_text}
<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>
"""
    elif template_type == 'qwen_instruct_with_legal_move_answer_only':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The answer must be enclosed within <answer> </answer> tags, respectively.
The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included.
Now, the user provides a FEN string and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
<|im_end|>
<|im_start|>user
Current FEN string: {fen}
{legal_moves_text}
<|im_end|>
<|im_start|>assistant\n<answer>
"""
    elif template_type == 'qwen_instruct_with_legal_move_eng_prompt':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included.
Now, the user provides a FEN string and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
The assistant's response must be provided exclusively in English.
<|im_end|>
<|im_start|>user
Current FEN string: {fen}
{legal_moves_text}
<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>
"""
    elif template_type == 'llama_instruct_wo_legal_move':
        prefix = f"""\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included.
Now, the user provides a FEN string.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
<|eot_id|><|start_header_id|>user<|end_header_id|>
Current FEN string: {fen}
<|eot_id|><|start_header_id|>assistant<|end_header_id|><|begin_of_text|>\nLet me solve this step by step.\n<think>
"""
    elif template_type == 'qwen_instruct_wo_legal_move':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in UCI notation, strictly using only the starting and ending square (e.g., e2e4, h7h6). No piece names, move types, or annotations should be included.
Now, the user provides a FEN string.
After analyzing the position, clearly state the best move in UCI notation within <answer> </answer> tags. i.e., <answer> e2e4 </answer>
<|im_end|>
<|im_start|>user
Current FEN string: {fen}
<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>
"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="./data/dev_chess")
    parser.add_argument('--data_path', default='./searchless_chess/data')
    parser.add_argument('--data_type', default='behavioral_cloning_data')
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--train_size', type=int, default=200000, help='actual train dataset size is 527633465')
    parser.add_argument('--test_size', type=int, default=1000, help='actual test dataset size is 62561')
    
    args = parser.parse_args()

    # Modify save_dir to include template type
    args.save_dir = args.save_dir + f"_{args.template_type}" + f"_{args.train_size}"+ f"_{args.test_size}"
    
    data_source = f'answer_chess_fen_{args.data_type}'
    train_data_path = f"{args.data_path}/train/{args.data_type}.bag"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size 

    def gen_from_bagz(path, limit=400000):  # Adjust limit as needed
        reader = bagz.BagReader(path)
        for i, data in enumerate(reader):
            if i >= limit:
                break
            fen, solution = constants.CODERS['behavioral_cloning'].decode(data)
            yield {'fen': fen, 'solution': solution}
    
    raw_train_dataset = Dataset.from_generator(gen_from_bagz, gen_kwargs={'path': train_data_path})
    print(len(raw_train_dataset))

    assert len(raw_train_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_train_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_train_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example['fen'], template_type=args.template_type)
            solution = {
                "fen": example['fen'],
                "next_move": example['solution'],
            }
            data = {
                "data_source": data_source,
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
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    save_dir = args.save_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

    train_filename = "train.parquet"
    test_filename = "test.parquet"

    train_dataset.to_parquet(os.path.join(save_dir, train_filename))
    test_dataset.to_parquet(os.path.join(save_dir, test_filename))
