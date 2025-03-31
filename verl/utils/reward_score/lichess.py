import re
import random
import chess
import time
import torch
import numpy as np

from typing import Dict, Tuple, Optional
from searchless_chess.src.engines import engine


def extract_solution(solution_str: str, logs: dict) -> Tuple[Optional[str], str, dict]:
    """Extract the final answer from the model's response string by isolating the assistant's output.
    
    Args:
        solution_str: Raw response string from the language model
        logs: Dictionary to store logging metrics.
        
    Returns:
        Tuple containing (extracted_answer, processed_string, logs)
    """
    header_markers = [
        "Assistant:",
        "<|start_header_id|>assistant<|end_header_id|><|begin_of_text|>",
        "<|im_start|>assistant"
    ]
    processed_str = None
    for marker in header_markers:
        if marker in solution_str:
            processed_str = solution_str.split(marker, 1)[1]
            break
    if processed_str is None:
        print("[Error] Failed to locate model response header")
        return None, solution_str, logs

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        return None, processed_str, logs
    else:
        logs['format'] = 1
        final_answer = matches[-1].group(1).strip()
        return final_answer, processed_str, logs

def validate_response_structure(processed_str: str, logs: dict) -> Tuple[bool, dict]:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        logs: Dictionary to store logging metrics.
        
    Returns:
        Tuple (validation_passed, logs)
    """
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = processed_str.find(tag_str)
        if count != expected_count:
            validation_passed = False
            logs['format'] = 0
            return validation_passed, logs

    # Verify tag order: <think> ... </think> ... <answer> ... </answer>
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        validation_passed = False
        logs['format'] = 0
    else:
        logs['format'] = 1
    
    return validation_passed, logs

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

def validate_chess_move_qvalue(
    fen: str,
    move: str,
    chess_model_qvalues: np.array,
    legal_moves: Tuple[str],
    logs: dict,
) -> Tuple[float, dict]:
    """
    Validates a chess move for a given FEN string and assigns a score based on the validation rules.
    
    Args:
        fen (str): The FEN string representing the board state.
        move (str): The move in SAN notation to validate.
        chess_model_qvalues (np.array): Numpy array with qvalues for each possible move
        legal_moves (List[str]): The legal moves in SAN notation.
        logs (dict): Dictionary to store logging metrics.
        
    Returns:
        Tuple containing (use_answer_reward, qvalue_reward, logs)
    """
    board = chess.Board(fen)
    use_answer_reward = False

    # strong penalty for
    # 1. not following san format
    # 2. following san format but illegal move
    try:
        move_uci  = board.parse_san(move).uci()
    except ValueError:
        return use_answer_reward, 0.0, logs
    
    if move not in legal_moves:
        return use_answer_reward, 0.0, logs
    else:
        logs['legal_move'] = 1

    # Obtain legal moves in the order used by the engine.
    legal_moves_ordered = [m.uci() for m in engine.get_ordered_legal_moves(board)]
    selected_index = legal_moves_ordered.index(move_uci)
    move_qvalue = chess_model_qvalues[selected_index] # qvalue of my selected move
    logs['q_value'] = move_qvalue
    
    sorted_win_indices = np.argsort(chess_model_qvalues)[::-1]
    optimal_move = legal_moves_ordered[sorted_win_indices[0]] # optimal move for current board
    if move_uci == optimal_move:
        use_answer_reward = True
        logs = _update_optimal_logs(len(legal_moves), logs)

    # compute regret
    optimal_move_qvalue = chess_model_qvalues[sorted_win_indices[0]] # qvalue of optimal move for current board
    logs['optimal_q_value_gap'] = move_qvalue - optimal_move_qvalue

    # the rank of our selected move
    move_rank = int(np.where(sorted_win_indices == selected_index)[0][0]) + 1 
    # Normalizes the rank into a score between 0 and 1, where 1 means we've done well, and 0 means we've done really bad
    logs['normalized_rank'] = (chess_model_qvalues.shape[0] - move_rank)/chess_model_qvalues.shape[0]
    
    return use_answer_reward, move_qvalue, logs

def validate_english_text(text: str, lg_detector: torch.nn.Module, logs: dict, threshold: float = 0.9) -> Tuple[float, dict]:
    """
    Validates that the generated text contains only ASCII characters.
    
    Args:
        text (str): The generated text.
        lg_detector (torch.nn.Module): Pytorch model which detects the language of the given text
        logs (dict): Dictionary to store logging metrics.
        threshold (float): Threshold for deciding whether the current text is english or not
        
    Returns:
        Tuple containing (use_english_reward, logs)
    """
    use_english_reward = False

    # Process the text
    with torch.inference_mode():
        doc = lg_detector(text)
        
    # The language detector returns a dictionary with keys 'language' and 'score'
    detected_lang = doc._.language
    score = doc._.language_score

    if detected_lang == "en" and score >= threshold:
        use_english_reward = True
    else:
        use_english_reward = False
    logs['english'] = use_english_reward

    return use_english_reward, logs

def compute_score(
    solution_str: str,
    ground_truth_dict: Dict[str, str],
    lg_detector: torch.nn.Module,
    chess_model_qvalues: np.array,
    format_reward: float = 0.1,
    english_reward: float = 0.1,
    answer_reward: float = 1.0,
    qvalue_reward_scaler: float = 1.0,
) -> float:
    """
    Computes comprehensive score for model response.
    """

    # Parse ground truth data
    board_fen = ground_truth_dict['board_fen']
    next_move_san = ground_truth_dict['next_move_san']
    legal_moves_san = ground_truth_dict['legal_moves_san']

    # Initialize log metrics
    logs = {'format': 0, 'legal_move': 0, 'english': 0, 
            'optimal': 0, 'optimal/1_10_moves': 0, 'optimal/11_20_moves': 0, 
            'optimal/21_30_moves': 0, 'optimal/31_40_moves': 0, 'optimal/over40_moves': 0}

    # Extract model answer (pass logs to the function)
    answer_text, processed_str, logs = extract_solution(solution_str, logs)
    
    # Validate response structure
    format_correct, logs = validate_response_structure(processed_str, logs)
    use_format_reward = True if format_correct else False

    # Validate English text
    use_english_reward = False
    if answer_text:
        use_english_reward, logs = validate_english_text(
            processed_str, 
            lg_detector,
            logs,
            threshold=0.9,
        )

    try:
        if format_correct and answer_text:
            use_answer_reward, qvalue_reward, logs = validate_chess_move_qvalue(
                                    fen=board_fen,
                                    move=answer_text,
                                    chess_model_qvalues=chess_model_qvalues,
                                    legal_moves=legal_moves_san,
                                    logs=logs)
        else:
            use_answer_reward = False
            qvalue_reward = 0.0
    except Exception as e:
        use_answer_reward = False
        qvalue_reward = 0.0
        print("\n[Content Validation] Skipped due to format errors or missing answer:", e)

    total_reward = use_format_reward * format_reward + \
                    use_english_reward * english_reward + \
                    use_answer_reward * answer_reward + \
                    qvalue_reward_scaler * qvalue_reward

    return total_reward, use_answer_reward, logs
