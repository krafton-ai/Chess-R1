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
    done = False
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
            if not done:
                san_tokens.append('which aims to')
                done = True
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
The user provides a FEN string representing the current board state. 
Your task is to select a best move with reasoning.

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

            if answer_uci == move_a_uci:
                answer_tactic = tactic_a
                answer_strategy = move_a_strategy
            else:
                answer_tactic = tactic_b
                answer_strategy = move_b_strategy

            # -----------------------------------------------
            # 4) Convert MoveA, MoveB, and final answer to SAN
            # -----------------------------------------------
            answer_obj = chess.Move.from_uci(answer_uci)
            answer_san = board.san(answer_obj)

            # -----------------------------------------------
            # 5) Convert Tactics to SAN
            # -----------------------------------------------
            tactic_san = convert_uci_list_to_san(answer_tactic, board)

            # -----------------------------------------------
            # 6) Construct Reasoning
            # -----------------------------------------------
            cot_reasoning = f"""<think> \
Given the position described by the FEN: {fen}. \
{answer_san} is a move that aims to {answer_strategy.lower()} This may lead to follow-up moves such as {tactic_san}.
Considering strategy and possible future outcomes, the best move is {answer_san}</think>
<answer>{answer_san}</answer>\
"""
            # Creating the transformed data entry
            transformed_entry = {
                "system": system,
                "instruction": instruction,
                "input": f'FEN: {fen}',
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
    parser.add_argument("--output_file", default="./processed_dataset/chess_best_move_14.json")
    args = parser.parse_args()

    transform_dataset(args.input_file, args.output_file)
