import re
import random
import chess
import time
import torch
import numpy as np

from typing import Dict, Tuple, Optional
from searchless_chess.src.engines import engine


def extract_solution(solution_str: str, logs: dict) -> Tuple[Optional[str], str, dict]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        logs: Dictionary to store logging metrics.
        
    Returns:
        Tuple containing (extracted_answer, processed_string, logs)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:  # base models
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|><|begin_of_text|>" in solution_str:  # Llama instruct
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|><|begin_of_text|>", 1)[1]
    elif "<|im_start|>assistant" in solution_str:  # Qwen instruct
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str, logs
    
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str, logs
    else:
        logs['format'] = 1
        final_answer = matches[-1].group(1).strip()
        return final_answer, processed_str, logs


def _compute_all_possible_actions() -> Tuple[dict[str, int], dict[int, str]]:
    """Returns two dicts converting moves to actions and actions to moves.
    
    These dicts contain all possible chess moves.
    """
    all_moves = []
    _CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    board = chess.BaseBoard.empty()
    for square in range(64):
        next_squares = []
        board.set_piece_at(square, chess.Piece.from_symbol('Q'))
        next_squares += board.attacks(square)
        board.set_piece_at(square, chess.Piece.from_symbol('N'))
        next_squares += board.attacks(square)
        board.remove_piece_at(square)
        for next_square in next_squares:
            all_moves.append(chess.square_name(square) + chess.square_name(next_square))
    
    # Handle promotions.
    promotion_moves = []
    for rank, next_rank in [('2', '1'), ('7', '8')]:
        for index_file, file in enumerate(_CHESS_FILE):
            move = f'{file}{rank}{file}{next_rank}'
            promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
            # Capture promotions: left side.
            if file > 'a':
                next_file = _CHESS_FILE[index_file - 1]
                move = f'{file}{rank}{next_file}{next_rank}'
                promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
            # Capture promotions: right side.
            if file < 'h':
                next_file = _CHESS_FILE[index_file + 1]
                move = f'{file}{rank}{next_file}{next_rank}'
                promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
    all_moves += promotion_moves

    move_to_action, action_to_move = {}, {}
    for action, move in enumerate(all_moves):
        assert move not in move_to_action
        move_to_action[move] = action
        action_to_move[action] = move

    return move_to_action, action_to_move

# Precompute all possible actions
MOVE_TO_ACTION_DICT, _ = _compute_all_possible_actions()

def _validate_move_format_and_legality(fen: str, move: str, logs: dict) -> Tuple[Optional[chess.Board], Optional[chess.Move], Optional[list]]:
    """
    Helper function to validate that a move is in proper UCI format and is legal on the board.
    
    Returns the board, the parsed move, and a sorted list of legal moves if valid.
    """
    if not re.match(r'^[a-h][1-8][a-h][1-8](?:[qrbn])?$', move):
        return None, None, None, logs
    logs['legal_uci'] = 1
    board = chess.Board(fen)
    try:
        parsed_move = board.parse_uci(move)
    except ValueError:
        return None, None, None, logs
    legal_moves = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION_DICT[x.uci()])
    if parsed_move not in legal_moves:
        print(f"Move {move} is not legal on the given board.")
        return None, None, None, logs
    logs['legal_move'] = 1

    return board, parsed_move, legal_moves, logs

def _update_optimal_logs(num_legal_moves: int, logs: dict) -> None:
    """
    Updates logs with counters based on the number of legal moves.
    """
    if num_legal_moves <= 10:
        logs['optimal/1_10_moves'] = 1
    elif num_legal_moves <= 20:
        logs['optimal/11_20_moves'] = 1
    elif num_legal_moves <= 30:
        logs['optimal/21_30_moves'] = 1
    elif num_legal_moves <= 40:
        logs['optimal/31_40_moves'] = 1
    else:
        logs['optimal/over40_moves'] = 1
    logs['optimal'] = 1

    return logs

def validate_chess_move_stockfish(
    fen: str, 
    move: str, 
    optimal_move: str, 
    logs: dict,
    success_reward: float = 1.0,
    weak_penalty_reward: float = -1.0,
    strong_penalty_reward: float = -2.0,
) -> Tuple[float, dict]:
    """
    Validates a chess move by comparing it against a provided optimal move (e.g. from Stockfish).
    """
    board, parsed_move, legal_moves, logs = _validate_move_format_and_legality(fen, move, logs)
    if board is None:
        return strong_penalty_reward, logs

    if move == optimal_move:
        logs = _update_optimal_logs(len(legal_moves), logs)
        return success_reward, logs
    else:
        return weak_penalty_reward, logs

def validate_chess_move_argmaxqvalue(
    fen: str, 
    move: str, 
    chess_model_qvalues: np.array,
    logs: dict,
    success_reward: float = 1.0,
    weak_penalty_reward: float = -1.0,
    strong_penalty_reward: float = -2.0,
) -> Tuple[float, dict]:
    """
    Validates a chess move using the chess model's analysis to determine optimality
    via the argmax of computed q-values.
    """
    board, parsed_move, legal_moves, logs = _validate_move_format_and_legality(fen, move, logs)
    if board is None:
        return strong_penalty_reward, logs
    
    sorted_win_indices = np.argsort(chess_model_qvalues)[::-1]
    
    # Obtain legal moves in the order used by the engine.
    legal_moves_ordered = [m.uci() for m in engine.get_ordered_legal_moves(board)]
    optimal_move = legal_moves_ordered[sorted_win_indices[0]]
    
    if move == optimal_move:
        logs = _update_optimal_logs(len(legal_moves), logs)
        return success_reward, logs
    else:
        return weak_penalty_reward, logs

def validate_chess_move_qvalue(
    fen: str, 
    move: str, 
    chess_model_qvalues: np.array,
    answer_scale: float,
    logs: dict,
    success_reward: float = 1.0,  # Not used since we return the q-value directly.
    weak_penalty_reward: float = -1.0,  # Not used here.
    strong_penalty_reward: float = -2.0,  # Used for error cases.
) -> Tuple[float, dict]:
    """
    Computes and returns the q-value for the selected move using the chess model's analysis.
    """
    board, parsed_move, legal_moves, logs = _validate_move_format_and_legality(fen, move, logs)
    if board is None:
        return strong_penalty_reward, logs
    
    legal_moves_ordered = [m.uci() for m in engine.get_ordered_legal_moves(board)]
    try:
        selected_index = legal_moves_ordered.index(move)
    except ValueError:
        print("Selected move not found in ordered legal moves.")
        return strong_penalty_reward, logs
    
    move_qvalue = chess_model_qvalues[selected_index] * answer_scale
    logs['q_value'] = move_qvalue

    # Also log optimal move information.
    sorted_win_indices = np.argsort(chess_model_qvalues)[::-1]
    optimal_move = legal_moves_ordered[sorted_win_indices[0]]
    if move == optimal_move:
        logs = _update_optimal_logs(len(legal_moves), logs)
    
    return move_qvalue, logs

def validate_english_text(text: str, lg_detector, logs: dict, english_reward: float = 1.0, non_english_penalty: float = -1.0, threshold: float = 0.9) -> Tuple[float, dict]:
    """
    Validates that the generated text contains only ASCII characters.
    
    Args:
        text (str): The generated text.
        logs (dict): Dictionary to store logging metrics.
        ascii_reward (float): Reward for text that is purely ASCII.
        non_ascii_penalty (float): Penalty if any non-ASCII characters are found.
        threshold (float): Threshold for deciding whether the current text is english or not
        
    Returns:
        Tuple containing the reward (or penalty) and updated logs.
    """
    # Process the text
    with torch.inference_mode():
        doc = lg_detector(text)
        
    # The language detector returns a dictionary with keys 'language' and 'score'
    detected_lang = doc._.language
    score = doc._.language_score

    if detected_lang == "en" and score >= threshold:
        logs['english'] = 1
        return english_reward, logs
    else:
        logs['english'] = 0
        return non_english_penalty, logs

def compute_score(solution_str: str, 
                 ground_truth_dict: Dict[str, str],
                 chess_model_qvalues,
                 answer_scale,
                 lg_detector,
                 use_data_gt,
                 use_chess_model_best,
                 use_chess_model_qvalue,
                 use_lg_consistency_rew,
                 format_reward: float = 0.1,
                 answer_reward: float = 2.0,
                 weak_penalty_reward: float = 0.0,
                 strong_penalty_reward: float = -2.0) -> float:
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string.
        ground_truth_dict: Dictionary containing ground truth data with keys 'fen' and 'next_move'.
        format_reward: Points awarded for correct format.
        answer_reward: Points awarded for a correct answer.
        
    Returns:
        Total score (sum of format and answer rewards)
    """

    # Parse ground truth data
    fen_state = ground_truth_dict.get('fen', '')
    solution_text = ground_truth_dict.get('next_move', '')

    # Initialize log metrics
    logs = {'format': 0, 'legal_uci': 0, 'legal_move': 0, 'english': 0, 'optimal': 0,
            'optimal/1_10_moves': 0, 'optimal/11_20_moves': 0, 'optimal/21_30_moves': 0, 
            'optimal/31_40_moves': 0, 'optimal/over40_moves': 0}

    # Extract model answer (pass logs to the function)
    answer_text, processed_str, logs = extract_solution(solution_str, logs)
    
    format_score = format_reward if logs['format']==1 else strong_penalty_reward

    # Validate English text
    english_score = 0
    if answer_text:
        english_score, logs = validate_english_text(
            processed_str, 
            lg_detector,
            logs, 
            english_reward=format_reward, 
            non_english_penalty=strong_penalty_reward,
            threshold=0.9,
        )
        if not use_lg_consistency_rew:
            english_score = 0

    # Validate answer content
    answer_score = 0
    num_legal_moves = 0
    
    try:
        if logs['format'] and answer_text:
            if use_data_gt:
                answer_score, logs = validate_chess_move_stockfish(
                                        fen=fen_state,
                                        move=answer_text,
                                        optimal_move=solution_text,
                                        logs=logs,
                                        success_reward=answer_reward,
                                        weak_penalty_reward=weak_penalty_reward,
                                        strong_penalty_reward=strong_penalty_reward)
            elif use_chess_model_best:
                answer_score, logs = validate_chess_move_argmaxqvalue(
                                        fen=fen_state,
                                        move=answer_text,
                                        chess_model_qvalues=chess_model_qvalues,
                                        logs=logs,
                                        success_reward=answer_reward,
                                        weak_penalty_reward=weak_penalty_reward,
                                        strong_penalty_reward=strong_penalty_reward)
            elif use_chess_model_qvalue:
                answer_score, logs = validate_chess_move_qvalue(
                                        fen=fen_state,
                                        move=answer_text,
                                        chess_model_qvalues=chess_model_qvalues,
                                        answer_scale=answer_scale,
                                        logs=logs,
                                        success_reward=answer_reward,
                                        weak_penalty_reward=weak_penalty_reward,
                                        strong_penalty_reward=strong_penalty_reward)
    except Exception as e:
        print("\n[Content Validation] Skipped due to format errors or missing answer:", e)
        answer_score = strong_penalty_reward

    total_score = format_score + answer_score + english_score
    # do_print = random.randint(1, 64) == 1
    do_print = False
    if do_print:
        print("\n" + "="*80)
        print(" Processing New Sample ".center(80, '='))
        print(f"\n[Model Response]\n{solution_str}")
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score, logs
