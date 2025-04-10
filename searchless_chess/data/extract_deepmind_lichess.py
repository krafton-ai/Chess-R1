import pandas as pd
import chess.pgn
import csv
import json
import requests
import time
import io
import argparse
from datasets import Dataset

def gen_from_csv(path, limit=400000):
    """
    Generator that reads the CSV file and yields dictionaries.
    Assumes the CSV has at least the columns: PuzzleId, FEN, Rating.
    """
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        if i >= limit:
            break
        
        board_fen = row['FEN']
        board = chess.Board(board_fen)

        # Push the current move to update the board state.
        # In lichess, one move is performed before a quiz
        moves_uci = row['Moves'].split(' ')
        cur_move_uci = moves_uci[0]
        cur_move_obj = chess.Move.from_uci(cur_move_uci)
        board.push(cur_move_obj)

        # Calculate the next move's SAN based on the updated board.
        next_move_uci = moves_uci[1]
        next_move_obj = chess.Move.from_uci(next_move_uci)
        next_move_san = board.san(next_move_obj)

        # Generate legal moves in SAN notation from the updated board.
        legal_moves = list(board.legal_moves)
        legal_moves_san = " ".join(board.san(move) for move in legal_moves)

        # Create a piece table from the updated board.
        piece_table = []
        board_rows = str(board).splitlines()
        for row_index, board_row in enumerate(board_rows):
            rank = 8 - row_index
            squares = board_row.split()
            for col_index, square in enumerate(squares):
                file = chr(ord('a') + col_index)
                piece = square if square != '.' else '-'  # Replace '.' with '-' for empty squares
                piece_table.append(f"{file}{rank}:{piece}")
        piece_table = " ".join(piece_table)

        yield {
            'id': row['PuzzleId'], 
            'rating': row['Rating'],
            'board_fen': board.fen(),  
            'next_move_san': next_move_san,
            'legal_moves_san': legal_moves_san,
            'piece_table': piece_table,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default="./searchless_chess/data/train/deepmind_lichess.csv")
    parser.add_argument('--data_path', default='./searchless_chess/data/deepmind_lichess_raw.csv')
    parser.add_argument('--data_size', type=int, default=10000)
    
    args = parser.parse_args()

    # Create raw dataset from CSV.
    one_step_dataset = Dataset.from_generator(
        gen_from_csv, 
        gen_kwargs={
            'path': args.data_path,
            'limit': args.data_size,
        }
    )
    
    # Convert the final dataset to a pandas DataFrame and save to CSV.
    df = one_step_dataset.to_pandas()
    df.to_csv(args.save_path, index=False)