""" Preprocess dataset for chess task """

import os
import argparse
import json
import pandas as pd

from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.think_chess import MOVE_TO_ACTION_DICT

import chess

def get_value(record, key):
    if key in record:
        return record[key]
    else:
        return ''

def make_prefix(template_type, chess_record):
    game_id = get_value(chess_record, 'id')
    rating = get_value(chess_record, 'rating')
    board_fen = get_value(chess_record, 'board_fen')
    board_str = get_value(chess_record, 'board_str')
    prev_moves_uci = get_value(chess_record, 'prev_moves_uci')
    prev_moves_san = get_value(chess_record, 'prev_moves_san')
    prev_moves_uci_numbering = get_value(chess_record, 'prev_moves_uci_numbering')
    prev_moves_san_numbering = get_value(chess_record, 'prev_moves_san_numbering')
    prev_moves_uci_numbering_space = get_value(chess_record, 'prev_moves_uci_numbering_space')
    prev_moves_san_numbering_space = get_value(chess_record, 'prev_moves_san_numbering_space')
    legal_moves_uci = get_value(chess_record, 'legal_moves_uci')
    legal_moves_san = get_value(chess_record, 'legal_moves_san')
    next_move_uci = get_value(chess_record, 'next_move_uci')
    next_move_san = get_value(chess_record, 'next_move_san')
    piece_table = get_value(chess_record, 'piece_table')

    if template_type == 'qwen_instruct_fen':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string.
After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>
<|im_end|>
<|im_start|>user
Current FEN string: {board_fen}
<|im_end|>
<|im_start|>assistant
Let's think step by step.
<think>
"""
    elif template_type == 'qwen_instruct_fen_legal':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string, and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>

<|im_end|>
<|im_start|>user
Current FEN string: {board_fen}
Legal moves: {legal_moves_san}
<|im_end|>
<|im_start|>assistant
Let's think step by step.
<think>
"""

    elif template_type == 'qwen_instruct_fen_legal_rule':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string, and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

<|im_end|>
<|im_start|>user
Current FEN string: {board_fen}
Legal moves: {legal_moves_san}
<|im_end|>
<|im_start|>assistant
Let's think step by step.
<think>
"""

    elif template_type == 'qwen_instruct_fen_legal_rule_table':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string, a list of legal moves for the given board, and a square by square chess piece position table.
After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

<|im_end|>
<|im_start|>user
Current FEN string: {board_fen}
Legal moves: {legal_moves_san}
Chess piece position table: {piece_table}
<|im_end|>
<|im_start|>assistant
Let's think step by step.
<think>
"""

    elif template_type == 'qwen_instruct_fen_legal_rule_table_pgn-full':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string, a list of legal moves for the given board, the PGN representing the game history up to the current FEN position, and a square by square chess piece position table.
After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

<|im_end|>
<|im_start|>user
Current FEN string: {board_fen}
Legal moves: {legal_moves_san}
PGN leading to FEN: {prev_moves_san_numbering}
Chess piece position table: {piece_table}
<|im_end|>
<|im_start|>assistant
Let's think step by step.
<think>
"""
    elif template_type == 'qwen_instruct_reastemp_fen_legal_rule':
        prefix = f"""\
<|im_start|>system
You are a helpful assistant who plays chess professionally.
The assistant first thinks through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how you analyze the position and decide on the best move, including:
  - A strategic evaluation of the position.
  - A comparison of key candidate moves.
  - For each candidate, consider the opponent's likely response and outcome.
  - Conclude with a clear justification for your final choice.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string, and a list of legal moves for the given board.
After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

<|im_end|>
<|im_start|>user
Current FEN string: {board_fen}
Legal moves: {legal_moves_san}
<|im_end|>
<|im_start|>assistant
Let's think step by step.
<think>
"""

    elif template_type == 'base_fen_legal_rule':
        """This works for any base model"""
        prefix = f"""\
A conversation between User and Assistant.
The User asks the best move to make for a given chess board state, and the Assistant solves it.
The Assistant is a professional chess player who first thinks about the reasoning process in the mind and then provides the User with the answer.
The Assistant’s reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
The reasoning process should describe how the Assistant analyzes the position and decide on the best move, including:
  - A strategic evaluation of the position.
  - A comparison of key candidate moves.
  - For each candidate, consider the opponent's likely response and outcome.
  - Conclude with a clear justification for your final choice.
The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Now, the user provides a FEN string and a list of legal moves for the given board.
After analyzing the position given by the user, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

User: The current FEN string is {board_fen} and legal moves are {legal_moves_san}. What is the best move to make out of the list of legal moves?

Assistant: \nLet me solve this step by step.\n<think>
"""
    else:
        raise NotImplementedError

    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="./data/lichess_db_puzzle_processed")
    parser.add_argument('--data_type', default='pgn')
    parser.add_argument('--train_data_path', default='./raw_data/lichess_db_puzzle_processed.csv')
    parser.add_argument('--test_data_path', default='./raw_data/lichess_db_puzzle_processed.csv')
    parser.add_argument('--template_type', type=str, default='qwen_instruct_reastemp_fen_legal_rule')

    args = parser.parse_args()

    args.save_dir = args.save_dir + f"_{args.template_type}"

    # We use only odd number data (our turn data)
    def gen_from_csv(path):
        df = pd.read_csv(path)
        id_counts = defaultdict(int)
        for row in df.to_dict(orient='records'):
            cid = row['id']
            # id_counts[cid] is the zero-based index of this row for that id
            if id_counts[cid] % 2 == 0:
                yield row
            id_counts[cid] += 1

    raw_train_dataset = Dataset.from_generator(gen_from_csv, gen_kwargs={'path': args.train_data_path})
    print(len(raw_train_dataset))

    TRAIN_SIZE = 100000
    TEST_SIZE = 5000

    assert len(raw_train_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_train_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_train_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": 'lichess',
                "prompt": [{
                    "role": "user",
                    "content": make_prefix(
                        template_type=args.template_type, chess_record=example
                    ),
                }],
                "ability": "chess",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "board_fen": example['board_fen'],
                        "next_move_san": example['next_move_san'],
                        "legal_moves_san": example['legal_moves_san'],
                    }
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