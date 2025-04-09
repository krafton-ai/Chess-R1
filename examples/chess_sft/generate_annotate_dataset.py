import json
import argparse
import re
import io
import wordninja
import tqdm
import chess.pgn


def normalize_spacing(text: str) -> str:
    """
    1) Reduce runs of whitespace to a single space.
    2) Ensure punctuation is followed by a space if there's none.
    3) Trim leading and trailing space.
    """
    # a) Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # b) Insert space if punctuation (.,!?;:) is immediately followed by a letter/number
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)

    # c) Trim leading and trailing
    text = text.strip()
    return text
    

def fix_compound_words(text: str) -> str:
    """
    Use wordninja to split 'compound' words lacking spaces.
    Example: 'muchprefer' -> 'much prefer'
    """
    tokens = text.split()
    corrected_tokens = []
    for token in tokens:
        # If the token is fairly long, attempt a dictionary-based split
        if len(token) > 6:
            subwords = wordninja.split(token)
            if len(subwords) > 1:
                corrected_tokens.append(" ".join(subwords))
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)
    return " ".join(corrected_tokens)


def remove_repeated_dots(text: str, keep_single_dot: bool = True) -> str:
    """
    Find repeated dots (like "..", "...", ". . .") and either collapse them
    down to one period, or remove them entirely.
    
    If keep_single_dot = True, then "...", ". . .", etc. all become "."
    If keep_single_dot = False, then repeated dots get removed entirely.
    """
    if keep_single_dot:
        # Replace sequences of 2+ periods (with optional space) by a single period and a space.
        # Example: "...", ". . ." -> ". "
        text = re.sub(r'(\.\s*){2,}', '. ', text)
    else:
        # Remove them entirely.
        text = re.sub(r'(\.\s*){2,}', '', text)
    # Trim extra spaces from that operation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_move_annotation_pairs(game: chess.pgn.Game, min_annotation_len):
    """
    Given a python-chess Game object, walk through each move.
    For each move that has a comment with length >= min_annotation_len,
    yield a dict with:
        {
          'fen': (FEN before this move),
          'move_san': (this move in SAN),
          'comment': (comment text)
        }
    """
    node = game
    move_comment_pairs = []

    while node.variations:
        next_node = node.variation(0)       # the next move
        move = next_node.move              # chess.Move
        comment = next_node.comment.strip() # comment after that move
        # The board before playing 'move'
        board_before = node.board()

        if move not in board_before.legal_moves:
            return []

        if len(comment) >= min_annotation_len:
            move_san = board_before.san(move)
            fen_before = board_before.fen()

            # Clean up the comment
            comment_clean = normalize_spacing(comment)
            comment_clean = fix_compound_words(comment_clean)
            comment_clean = remove_repeated_dots(comment_clean)

            pair_data = {
                "fen": fen_before,
                "move_san": move_san,
                "comment": comment_clean
            }
            move_comment_pairs.append(pair_data)

        # Advance
        node = next_node

    return move_comment_pairs


def transform_dataset(input_file, output_file, min_annotation_len):
    """
    Transforms the dataset to the desired format for fine-tuning Qwen2.5.

    Args:
        input_file (str): Path to the input JSONL file containing the original dataset.
        output_file (str): Path to the output JSONL file to save the transformed dataset.
    """
    transformed_entries = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm.tqdm(infile):
            data = json.loads(line)

            # -----------------------------
            # 1) Construct the instruction
            # -----------------------------
            system = """You are a professional chess assistant."""
            instruction = """\
The user provides a FEN string representing the current board state and a single candidate move. 
Your task is to explain or annotate the given move in detail, addressing any strategic ideas or tactical elements you observe.
"""

            # ------------------------------------------------
            # 2) Remove annotation not extracted from gameknot
            # -------------------------------------------------
            pgn_text = data.get("text", "")

            # Check if this game is from gameknot.com
            # i.e., there is a line: [Site "http://gameknot.com/"]
            if '[Site "http://gameknot.com/"]' not in pgn_text:
                continue
            
            # ------------------------------------------------
            # 2) Filter texts except content
            # -------------------------------------------------

            # Split the PGN text into lines
            lines = pgn_text.split("\n")

            # Filter out lines containing [Event ...] or [Site ...]
            cleaned_lines = []
            for l in lines:
                # If the line starts with [Event or [Site, skip it
                if (l.startswith("[Event") 
                    or l.startswith("[Site")
                    or l.startswith("[Date")
                    or l.startswith("[Round")
                    or l.startswith("[White")
                    or l.startswith("[Black")
                    or l.startswith("[Result")
                    or l.startswith("[TimeControl")
                ):
                    continue
                cleaned_lines.append(l)

            annotations = " ".join(cleaned_lines)   

            # ------------------------------------------------
            # 3) Clean up the spaces and compounds
            # -------------------------------------------------
            annotations = normalize_spacing(annotations)
            annotations = fix_compound_words(annotations)

            # ------------------------------------------------
            # 4) Extract fen, move, annotation pairs
            # -------------------------------------------------
            pgn_io = io.StringIO(annotations)
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                continue

            move_annotation_data = generate_move_annotation_pairs(game, min_annotation_len=min_annotation_len)

            # -----------------------------------------------
            # 5) Construct Dataset
            # -----------------------------------------------
            # For each valid move, create an instruction entry
            for item in move_annotation_data:
                fen_before = item["fen"]
                move_san = item["move_san"]
                annotation = item["comment"]

                # Our "instruction" can be a standard prompt
                # Our "input" includes the FEN plus the candidate move
                # Our "output" is the move’s commentary
                entry = {
                    "system": system,
                    "instruction": instruction,
                    "input": f"FEN: {fen_before} Move: {move_san}",
                    "output": f"Given the position: {fen_before}, we are evaluating a move: {move_san}. \
{move_san} is annotated as {annotation}"
                }
                transformed_entries.append(entry)

    # Writing all transformed entries to the output file at once
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in transformed_entries:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform dataset for Qwen2.5 fine-tuning.")
    parser.add_argument("--input_file",  default="./raw_dataset/annotated_pgn-data.jsonl-00000-of-00002")
    parser.add_argument("--output_file", default="./processed_dataset/chess_annotation_0.json")
    parser.add_argument("--min_annotation_len", default=100)
    args = parser.parse_args()

    transform_dataset(args.input_file, args.output_file, args.min_annotation_len)
