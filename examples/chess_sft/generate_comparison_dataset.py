import json
import argparse
import chess.pgn


def convert_uci_list_to_san(uci_string, board):
    """
    Given a string of space-separated tokens, try to convert each token
    from UCI to SAN by pushing it on 'board'. If a token is not a valid
    UCI move, fall back to using the original token.
    Returns the resulting string (space-separated).
    """
    tokens = uci_string.split()
    san_tokens = []
    for token in tokens:
        # If there's punctuation like "Checkmate!" or "?", you may want
        # to strip it or handle it differently. Here we do a simple try/except.
        try:
            move_obj = chess.Move.from_uci(token)
            if move_obj in board.legal_moves:
                san_move = board.san(move_obj)
                board.push(move_obj)
                san_tokens.append(san_move)
            else:
                # Not a legal move in the current position
                san_tokens.append(token)
        except ValueError:
            # Not a valid UCI string, e.g. "Checkmate!"
            san_tokens.append(token)
    result = " ".join(san_tokens)
    result = result.rstrip('.')

    return result


def transform_dataset(input_file, output_file):
    """
    Transforms the dataset to the desired format for fine-tuning Qwen2.5.

    Args:
        input_file (str): Path to the input JSONL file containing the original dataset.
        output_file (str): Path to the output JSONL file to save the transformed dataset.
    """
    transformed_data = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)

            # -----------------------------
            # 1) Construct the instruction
            # -----------------------------
            system = """You are a professional chess assistant."""
            instruction = """\
The user provides a FEN string representing the current board state and two candidate moves. 
Your task is to analyze both moves through reasoning and determine the superior one.

1. During reasoning, enclose your detailed thought process inside <think> </think> tags. This should include:
- A strategic evaluation of the position.
- A comparison of key candidate moves with potential follow-up plans and long-term consequences of each move.
2. After finishing your analysis, state the better move inside <answer> </answer> tags using SAN notation (e.g., Nf3, Qxe6, g8=Q)."""

            # -----------------------------
            # 2) Extract the FEN
            # -----------------------------            
            fen_start = data["input"].find('"') + 1
            fen_end = data["input"].find('"', fen_start)
            fen = data["input"][fen_start:fen_end]

            # Initialize the chess board from the FEN
            board = chess.Board(fen)

            # -------------------------------------
            # 3) Extract MoveA and MoveB (in UCI)
            # -------------------------------------
            move_a_start = data["input"].find("MoveA:")
            move_a_end = data["input"].find("MoveB:")
            move_a_section = data["input"][move_a_start:move_a_end].strip()
            move_a_lines = move_a_section.split("TacticA:")
            move_a_full = move_a_lines[0].replace("MoveA:", "").strip()
            tactic_a = move_a_lines[1].strip() if len(move_a_lines) > 1 else ""
            move_a_uci = move_a_full.split(",", 1)[0].strip()
            move_a_strategy = move_a_full.split(",", 1)[-1].strip() if "," in move_a_full else ""

            # Extract MoveB
            move_b_section = data["input"][move_a_end:].strip()
            move_b_lines = move_b_section.split("TacticB:")
            move_b_full = move_b_lines[0].replace("MoveB:", "").strip()
            tactic_b = move_b_lines[1].strip() if len(move_b_lines) > 1 else ""
            move_b_uci = move_b_full.split(",", 1)[0].strip()
            move_b_strategy = move_b_full.split(",", 1)[-1].strip() if "," in move_b_full else ""

            # Extract final answer
            answer_uci = data["output"].split(":")[-1].strip()

            # -----------------------------------------------
            # 4) Convert MoveA, MoveB, and final answer to SAN
            # -----------------------------------------------
            try:
                move_a_obj = chess.Move.from_uci(move_a_uci)
                move_a_san = board.san(move_a_obj)
                move_b_obj = chess.Move.from_uci(move_b_uci)
                move_b_san = board.san(move_b_obj)
                answer_obj = chess.Move.from_uci(answer_uci)
                answer_san = board.san(answer_obj)

            except ValueError:
                print('cannot convert uci to san')
                move_a_san = move_a_uci
                move_b_san = move_b_uci
                answer_san = answer_uci

            # -----------------------------------------------
            # 5) Convert Tactics to SAN
            # -----------------------------------------------
            board_a = board.copy()
            tactic_a_san = convert_uci_list_to_san(tactic_a, board_a)

            board_b = board.copy()
            tactic_b_san = convert_uci_list_to_san(tactic_b, board_b)

            # -----------------------------------------------
            # 6) Construct Reasoning
            # -----------------------------------------------
            cot_reasoning = f"""<think> \
Given the position described by the FEN: {fen}, we are evaluating two candidate moves. \
{move_a_san} is a move that aims to {move_a_strategy.lower()} This may lead to follow-up moves such as {tactic_a_san}.
In contrast, {move_b_san} aims to {move_b_strategy.lower()} This can potentially resulting in sequences like {tactic_b_san}.
After comparing both plans in terms of strategy and possible future outcomes, the better move is {answer_san}</think>
<answer>{answer_san}</answer>\
"""
            # Creating the transformed data entry
            transformed_entry = {
                "system": system,
                "instruction": instruction,
                "input": f'FEN: {fen}, move1: {move_a_san}, move2: {move_b_san}',
                "output": cot_reasoning,
            }

            transformed_data.append(transformed_entry)

    # Writing all transformed entries to the output file at once
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in transformed_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform dataset for Qwen2.5 fine-tuning.")
    parser.add_argument("--input_file",  default="./raw_dataset/explain_dataset14_both.jsonl")
    parser.add_argument("--output_file", default="./processed_dataset/chess_comparison_14.json")
    args = parser.parse_args()

    transform_dataset(args.input_file, args.output_file)
