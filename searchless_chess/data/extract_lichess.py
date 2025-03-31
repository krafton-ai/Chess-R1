import pandas as pd
import chess.pgn
import csv
import json
import requests
import time
import io
import argparse
from datasets import Dataset

def get_lichess_puzzle(puzzleID):
    """
    Fetch puzzle info from Lichess API given a puzzle ID.
    """
    lichess_url = 'https://lichess.org/api/puzzle/' + str(puzzleID)
    response = requests.get(lichess_url)
    response_json = response.json()

    return response_json

def group_moves(moves_list, dot_space=True):
    """
    Given a list of moves (as strings), group them into full moves.
    If dot_space is True, a dot and a space are inserted (e.g., "1. e4 e5").
    If False, no space after the dot (e.g., "1.e4 e5").
    """
    groups = []
    for i in range(0, len(moves_list), 2):
        move_num = i // 2 + 1
        if dot_space:
            if i + 1 < len(moves_list):
                group = f"{move_num}. {moves_list[i]} {moves_list[i+1]}"
            else:
                group = f"{move_num}. {moves_list[i]}"
        else:
            if i + 1 < len(moves_list):
                group = f"{move_num}.{moves_list[i]} {moves_list[i+1]}"
            else:
                group = f"{move_num}.{moves_list[i]}"
        groups.append(group)
    return " ".join(groups)

def get_puzzle_moves_info(moves, ply, solution):
    """
    Extracts the moves leading to the puzzle board ("prev" moves) and the solution moves ("next" moves)
    in both UCI and SAN formats. Also computes numbered versions of the previous moves.
    
    Parameters:
      moves (str): Full PGN moves string of the game.
      ply (int): The puzzle's initial ply (number of half-moves) to reach the puzzle board.
      solution (list): List of UCI moves for the solution.
      
    Returns:
      dict: Contains:
         "board_fen": FEN string after playing the moves leading to the puzzle.
         "prev_moves_uci": Space-separated string of moves (UCI format) leading to the puzzle board.
         "prev_moves_san": Space-separated string of moves (SAN format) leading to the puzzle board.
         "prev_moves_uci_numbering": Numbered UCI moves with no space after the dot.
         "prev_moves_san_numbering": Numbered SAN moves with no space after the dot.
         "prev_moves_uci_numbering_space": Numbered UCI moves with a space after the dot.
         "prev_moves_san_numbering_space": Numbered SAN moves with a space after the dot.
         "next_moves_uci": List of solution moves in UCI format.
         "next_moves_san": List of solution moves in SAN format.
    """
    game = chess.pgn.read_game(io.StringIO(moves))
    board = game.board()
    prev_moves_uci = []
    prev_moves_san = []
    
    # Play moves until the specified ply is reached.
    for i, move in enumerate(game.mainline_moves()):
        if i > ply:
            break
        uci_move = move.uci()
        san_move = board.san(move)
        prev_moves_uci.append(uci_move)
        prev_moves_san.append(san_move)
        board.push(move)

    board_fen = board.fen()
    
    # Group previous moves into full moves (using chess turn indexing).
    prev_moves_uci_numbering = group_moves(prev_moves_uci, dot_space=False)
    prev_moves_san_numbering = group_moves(prev_moves_san, dot_space=False)
    prev_moves_uci_numbering_space = group_moves(prev_moves_uci, dot_space=True)
    prev_moves_san_numbering_space = group_moves(prev_moves_san, dot_space=True)
    
    # Compute solution moves: keep them as a list.
    next_moves_uci = solution if isinstance(solution, list) else [solution]
    
    # Convert each next move to SAN format using a copy of the board.
    board_for_solution = board.copy()
    next_moves_san = []
    for move_uci in next_moves_uci:
        move_obj = chess.Move.from_uci(move_uci)
        try:
            san_move = board_for_solution.san(move_obj)
        except:
            san_move = f"illegal_move({move_uci})"
        next_moves_san.append(san_move)
        board_for_solution.push(move_obj)
    
    return {
         "board_fen": board_fen,
         "prev_moves_uci": " ".join(prev_moves_uci),
         "prev_moves_san": " ".join(prev_moves_san),
         "prev_moves_uci_numbering": prev_moves_uci_numbering,
         "prev_moves_san_numbering": prev_moves_san_numbering,
         "prev_moves_uci_numbering_space": prev_moves_uci_numbering_space,
         "prev_moves_san_numbering_space": prev_moves_san_numbering_space,
         "next_moves_uci": next_moves_uci,
         "next_moves_san": next_moves_san,
    }

def add_moves_info_columns(example):
    """
    For each record in the raw dataset (which already has 'id' and 'rating'),
    fetch the puzzle details from the Lichess API and extract:
      - The board position (board_fen) reached after playing the moves leading to the puzzle.
      - The moves leading to the puzzle ("prev" moves) in both UCI and SAN formats,
        including numbered versions.
      - The solution moves ("next" moves) in both UCI and SAN formats.
    Returns a dict with these new fields.
    """
    puzzleID = example["id"]
    try:
        puzzle_dict = get_lichess_puzzle(puzzleID)
        ply = int(puzzle_dict["puzzle"]["initialPly"])
        moves = puzzle_dict["game"]["pgn"]
        solution = puzzle_dict["puzzle"]["solution"]
        moves_info = get_puzzle_moves_info(moves, ply, solution)
    except:
        print(f"Failed processing puzzle {puzzleID}")
        moves_info = {
            "board_fen": '',
            "prev_moves_uci": '',
            "prev_moves_san": '',
            "prev_moves_uci_numbering": '',
            "prev_moves_san_numbering": '',
            "prev_moves_uci_numbering_space": '',
            "prev_moves_san_numbering_space": '',
            "next_moves_uci": [],  # return an empty list
            "next_moves_san": []   # return an empty list
        }
    return moves_info

def split_next_moves(example):
    """
    Given a puzzle record with fields:
      id, rating, board_fen, prev_moves_uci, prev_moves_san,
      prev_moves_uci_numbering, prev_moves_san_numbering,
      prev_moves_uci_numbering_space, prev_moves_san_numbering_space,
      next_moves_uci (list), next_moves_san (list),
    return a list of records where each record corresponds to one next move (one-step move).

    For each action:
      - Start from the board state (from board_fen) and push the next move to get an updated FEN.
      - Cumulatively append the move to the previous moves (for all formats).
      - Each record contains the updated board_fen, cumulative prev_moves (all formats),
        the one-step next move, and forwards the original "id" and "rating".
    """
    if not example.get("next_moves_uci"):
        return []

    if len(example['next_moves_uci']) == 0:
        return []
    
    records = []
    # Initialize the board from the puzzle board_fen.
    board = chess.Board(example["board_fen"])
    # Start with cumulative prev moves.
    prev_moves_uci = example["prev_moves_uci"].split() if example["prev_moves_uci"] else []
    prev_moves_san = example["prev_moves_san"].split() if example["prev_moves_san"] else []
    
    for uci, san in zip(example["next_moves_uci"], example["next_moves_san"]):
        # Record the current board FEN before applying the move.
        current_board_fen = board.fen()
        current_legal_moves = list(board.legal_moves)
        current_legal_moves_uci = []
        current_legal_moves_san = []
        for legal_move in current_legal_moves:
            current_legal_moves_uci.append(board.uci(legal_move))
            current_legal_moves_san.append(board.san(legal_move))

        current_board_str = str(board).replace("\n", " \n ")

        # Recompute the numbering fields based on current cumulative moves.
        numbering_uci = group_moves(prev_moves_uci, dot_space=False)
        numbering_san = group_moves(prev_moves_san, dot_space=False)
        numbering_uci_space = group_moves(prev_moves_uci, dot_space=True)
        numbering_san_space = group_moves(prev_moves_san, dot_space=True)
        record = {
            "id": example["id"],
            "rating": example["rating"],
            "board_fen": current_board_fen,
            "board_str": current_board_str,
            "prev_moves_uci": " ".join(prev_moves_uci),
            "prev_moves_san": " ".join(prev_moves_san),
            "prev_moves_uci_numbering": numbering_uci,
            "prev_moves_san_numbering": numbering_san,
            "prev_moves_uci_numbering_space": numbering_uci_space,
            "prev_moves_san_numbering_space": numbering_san_space,
            "legal_moves_uci": " ".join(current_legal_moves_uci),
            "legal_moves_san": " ".join(current_legal_moves_san),
            "next_move_uci": uci,
            "next_move_san": san,
        }
        records.append(record)
        
        # Now push the candidate next move to update the board and cumulative lists.
        move = chess.Move.from_uci(uci)
        try:
            board.push(move)
        except:
            print(f"Error pushing move {uci}: {e}")
            break
        prev_moves_uci.append(uci)
        prev_moves_san.append(san)
    
    return records

def gen_from_csv(path, limit=400000):
    """
    Generator that reads the CSV file and yields dictionaries.
    Assumes the CSV has at least the columns: PuzzleId, FEN, Rating.
    """
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        if i >= limit:
            break
        yield {
            'id': row['PuzzleId'], 
            'rating': row['Rating'],
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default="./searchless_chess/data/test/puzzles.csv")
    parser.add_argument('--data_path', default='./searchless_chess/data/puzzles.csv')
    parser.add_argument('--data_size', type=int, default=10000)
    
    args = parser.parse_args()

    # Create raw dataset from CSV.
    raw_dataset = Dataset.from_generator(
        gen_from_csv, 
        gen_kwargs={
            'path': args.data_path,
            'limit': args.data_size,
        }
    )
    
    # Enrich each puzzle with move details.
    enriched_dataset = raw_dataset.map(add_moves_info_columns)
    
    # Flatten each puzzle record into one record per next move (with cumulative prev moves and updated FEN).
    one_step_records = []
    for example in enriched_dataset:
        one_step_records.extend(split_next_moves(example))    

    one_step_dataset = Dataset.from_list(one_step_records)
    
    # Convert the final dataset to a pandas DataFrame and save to CSV.
    df = one_step_dataset.to_pandas()
    df.to_csv(args.save_path, index=False)