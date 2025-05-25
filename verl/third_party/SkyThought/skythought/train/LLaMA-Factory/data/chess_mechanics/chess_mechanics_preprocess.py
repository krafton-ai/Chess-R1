# How many different tasks there are and what are their counts?
import chess
import json
import random
import re
from collections import Counter
from tqdm import tqdm

def simulate_pgn_segment(pgn_text: str):
    """
    Given a PGN string of a chess game, this function:
      1. Extracts all moves (in SAN format) from the PGN.
      2. Randomly selects a starting index among the moves.
      3. Plays the game from the start until that point and computes the intermediate FEN.
      4. Randomly selects between 1 and 5 moves (subject to moves remaining) starting from that index,
         and then applies them to the board.
      5. Generates a natural chain-of-thought reasoning trace that explains, step by step, how
         each move changes the board. For the first move, the current FEN is explicitly mentioned;
         for subsequent moves, the explanation is more concise.
         The reasoning is enclosed in <think>...</think> tags.
      6. Returns:
          - A prompt string (which includes the starting FEN and summary of applied moves),
          - The list of applied SAN moves,
          - The resulting FEN after applying those moves,
          - The natural-language chain-of-thought reasoning trace.
    
    Parameters:
        pgn_text (str): The PGN string including move numbers 
                        (e.g., "1. e4 g6 2. d4 Bg7 3. c4 c5 4. Nc3 cxd4 ...").
    
    Returns:
        tuple: (prompt, applied_moves, result_fen, reasoning_trace)
    """
    think_action_tags = f"Use <think>...</think> tags to explain your reasoning and <answer>...</answer> tags to give the final FEN."
    system_prompt = f"Generate the resulting FEN position given the following SAN moves, starting from the initial FEN. {think_action_tags}"

    # Define a regex pattern to capture moves.
    move_pattern = r'\d+\.\s*([^\s]+)(?:\s+([^\s]+))?'
    move_matches = re.findall(move_pattern, pgn_text)
    
    # Flatten the matches into a list of moves (stripping any trailing period)
    pgn_moves = []
    for first, second in move_matches:
        if first:
            pgn_moves.append(first.rstrip('.?'))
        if second:
            pgn_moves.append(second.rstrip('.?'))
    
    if not pgn_moves:
        raise ValueError("No moves extracted from the PGN text.")
    
    # Remove a trailing move that might be an annotation/result.
    pgn_moves = pgn_moves[:-1]
    
    # Create a board instance (using the standard initial position)
    board = chess.Board()
    
    # Randomly choose a starting index among all moves.
    start_index = random.randint(0, len(pgn_moves) - 1)
    
    # Play all moves up to the starting index to obtain the intermediate FEN.
    for move in pgn_moves[:start_index]:
        try:
            board.push_san(move)
        except Exception as e:
            raise ValueError(f"Error applying move '{move}' at index {pgn_moves.index(move)}: {e}")
    
    intermediate_fen = board.fen()
    starting_fen_str = f"The current FEN is {intermediate_fen}."
    
    # Determine how many moves we can apply from the remaining moves.
    remaining_moves = len(pgn_moves) - start_index
    num_moves_to_apply = random.randint(1, min(5, remaining_moves))
    
    # Extract the moves to apply.
    moves_to_apply = pgn_moves[start_index : start_index + num_moves_to_apply]
    
    # Build a natural chain-of-thought reasoning trace.
    reasoning_lines = []
    board_trace = board.copy()  # Copy the board for a step-by-step trace.

    # Define templates for the first move (7 examples) without using "I".
    first_move_templates = [
        "Starting from FEN {current_fen}, the move '{move}' is played, resulting in {new_fen}.",
        "From FEN {current_fen}, move '{move}' leads to {new_fen}.",
        "At FEN {current_fen}, the move '{move}' produces {new_fen}.",
        "Beginning at FEN {current_fen}, playing '{move}' updates the board to {new_fen}.",
        "FEN {current_fen} changes to {new_fen} after move '{move}' is executed.",
        "With the starting FEN at {current_fen}, move '{move}' transforms the board into {new_fen}.",
        "From the initial FEN {current_fen}, move '{move}' yields {new_fen}."
    ]

    # Define templates for subsequent moves (7 examples) without using "I".
    subsequent_move_templates = [
        "Then, move '{move}' is applied, resulting in {new_fen}.",
        "Following that, '{move}' changes the position to {new_fen}.",
        "Next, the move '{move}' updates the board to {new_fen}.",
        "Afterwards, '{move}' is executed, producing {new_fen}.",
        "Subsequently, move '{move}' yields {new_fen}.",
        "After move '{move}', the board becomes {new_fen}.",
        "Finally, applying '{move}' gives a new FEN: {new_fen}."
    ]
    
    # Build the reasoning trace move by move.
    for idx, move in enumerate(moves_to_apply):
        current_fen = board_trace.fen()
        try:
            board_trace.push_san(move)
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise ValueError(f"Error applying move '{move}' during trace: {e}")
        new_fen = board_trace.fen()
        if idx == 0:
            template = random.choice(first_move_templates)
            line = template.format(current_fen=current_fen, move=move, new_fen=new_fen)
        else:
            template = random.choice(subsequent_move_templates)
            line = template.format(move=move, new_fen=new_fen)
        reasoning_lines.append(line)
    
    # Wrap the reasoning trace in <think> tags.
    reasoning_trace = "<think>" + " ".join(reasoning_lines) + "</think>"
    
    moves_summary = ", ".join(moves_to_apply)
    moves_summary_full = f" And the next applied SAN moves are: {moves_summary}."

    # Build the system prompt string.
    user_prompt = f"{starting_fen_str}{moves_summary_full}"
    
    # Now, apply the selected moves to the original board (which is at the intermediate state) to get the final FEN.
    for move in moves_to_apply:
        try:
            board.push_san(move)
        except Exception as e:
            raise ValueError(f"Error applying move '{move}' from index {start_index}: {e}")
    result_fen = board.fen()
    answer_response = f" <answer>{result_fen}</answer>"

    assistant_prompt = reasoning_trace + answer_response

    return system_prompt, user_prompt, assistant_prompt


def generate_legal_moves_given_fen(text: str):
    """
    Splits the provided text into sentences so that:
      - All sentences except the final one are treated as the input prompt.
      - The final sentence is treated as the response.
    
    Returns:
        tuple: A tuple (prompt_text, response_text), where prompt_text is the concatenation
        of all sentences except the final one and response_text is the last sentence.
    """
    system_prompt = "Generate all legal moves in SAN format for the given FEN board state and enclose it within <answer>...</answer> tags."
    
    text = text.strip()
    
    # Use regex to find the FEN substring.
    fen_pattern = r"([rnbqkpRNBQKP1-8/]+\s+[wb]\s+[-KQkq]+\s+(?:[a-h][36]|-)\s+\d+\s+\d+)"
    fen_match = re.search(fen_pattern, text)
    if fen_match:
        # Reconstruct the FEN using all captured groups.
        fen_str = fen_match.group(1).strip()
    else:
        raise ValueError("No valid FEN found in the provided text.")
    
    # Create the user prompt using only the current FEN board state.
    user_prompt = f"Current FEN board state is {fen_str}."
    
    board = chess.Board(fen_str)
    legal_moves_san = [board.san(move) for move in board.legal_moves]
    legal_moves_san = (', ').join(legal_moves_san)

    # We assume the final sentence is the assistant's prompt.
    assistant_prompt = f"<answer>{legal_moves_san}</answer>"
    
    return system_prompt, user_prompt, assistant_prompt


def draw_board_given_fen(text: str):
    """
    Processes the provided text by:
      - Extracting the FEN string from the text.
      - Splitting the provided text into sentences so that:
          - All sentences except the final one are treated as the input prompt.
          - The final sentence is treated as the response.
    
    Returns:
        tuple: A tuple (fen, prompt_text, response_text), where 'fen' is the extracted FEN string,
               'prompt_text' is the concatenation of all sentences except the final one, and
               'response_text' is the final sentence.
    """

    system_prompt = "Given a FEN string, draw the current chessboard position and enclose it within <answer>...</answer> tags."

    # First, extract the FEN.
    # The FEN is expected to follow the standard structure:
    # [piece placement] [side to move] [castling availability] [en passant target square] [halfmove clock] [fullmove number]
    text = text.strip()
    
    # Robust FEN pattern:
    #   - The board part must have exactly 8 ranks separated by 7 slashes.
    #   - Followed by side to move (w or b), castling availability, en passant target, halfmove clock, and fullmove number.
    fen_pattern = (r'((?:[rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+)\s+'
                   r'([wb])\s+'
                   r'([-KQkq]+)\s+'
                   r'((?:[a-h][36])|-)\s+'
                   r'(\d+)\s+'
                   r'(\d+)')
    fen_match = re.search(fen_pattern, text)
    if not fen_match:
        raise ValueError("No valid FEN string found in the input text.")
    
    fen = fen_match.group(1)

    def textify_chess_board(fen: str):
        board = chess.Board(fen)
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
        return piece_table

    new_chess_board_str = textify_chess_board(fen)
    
    # Extract the substring between the period and the colon.
    user_prompt = f"Current FEN string is {fen}."

    assistant_prompt = f"<answer>{new_chess_board_str}</answer>"

    return system_prompt, user_prompt, assistant_prompt



# Mapping common to both training and testing
DESCRIPTION_TO_FUNC = {
    "Generate FEN given PGN": simulate_pgn_segment,
    "Draw chess board given FEN": draw_board_given_fen,
    "Generate all legal moves in SAN format given the board FEN": generate_legal_moves_given_fen,
}

def process_files(jsonl_files, output_filename, description_mapping):
    """
    Processes a list of jsonl files and writes the formatted output
    to the specified JSON file.
    """
    unique_descriptions = Counter()
    alpaca_data_list = []

    # Process each file
    for filename in tqdm(jsonl_files, desc="Processing files"):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc=f"Reading {filename}", leave=False):
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    description = data.get("metadata", {}).get("description", "")
                    text = data.get("text", "").strip()
                    if not text:
                        continue

                    # Retrieve processing function based on description
                    func = description_mapping.get(description)
                    if not func:
                        continue

                    unique_descriptions[description] += 1
                    system_prompt, user_prompt, assistant_prompt = func(text)

                    alpaca_data_list.append({
                        'instruction': system_prompt,
                        'input': user_prompt,
                        'output': assistant_prompt,
                        'system': "You are a professional chess assistant.",
                    })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Log counts
    total_samples = sum(unique_descriptions.values())
    print("Number of unique descriptions:", unique_descriptions)
    print("Total number of samples:", total_samples)

    # Shuffle the dataset
    random.shuffle(alpaca_data_list)

    # Save data to output file
    try:
        with open(output_filename, 'w', encoding='utf-8') as fout:
            json.dump(alpaca_data_list, fout, indent=2)
        print("Preprocessing complete. Data saved to:", output_filename)
    except Exception as e:
        print(f"Error writing to file {output_filename}: {e}")
    
    return alpaca_data_list

if __name__ == "__main__":
    # Process training files
    train_jsonl_files = [
        'chess_modeling-data.jsonl-00000-of-00008',
        'chess_modeling-data.jsonl-00001-of-00008',
        'chess_modeling-data.jsonl-00002-of-00008',
        'chess_modeling-data.jsonl-00003-of-00008',
        'chess_modeling-data.jsonl-00004-of-00008',
        'chess_modeling-data.jsonl-00005-of-00008',
        'chess_modeling-data.jsonl-00006-of-00008'
    ]
    process_files(train_jsonl_files, "chess_mechanics_train.json", DESCRIPTION_TO_FUNC)

    # Process test files
    test_jsonl_files = [
        'chess_modeling-data.jsonl-00007-of-00008'
    ]
    process_files(test_jsonl_files, "chess_mechanics_test.json", DESCRIPTION_TO_FUNC)