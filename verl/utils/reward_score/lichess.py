import re
import random
import chess
from typing import Dict, Tuple, Optional, Tuple


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

def validate_chess_move(
    fen: str,
    move: str,
    optimal_move: str,
    legal_moves: Tuple[str],
    logs: dict,
    success_reward: float = 1.0,
    weak_penalty_reward: float = -1.0,
    strong_penalty_reward: float = -2.0,
) -> Tuple[float, dict]:
    """
    Validates a chess move for a given FEN string and assigns a score based on the validation rules.
    
    Args:
        fen (str): The FEN string representing the board state.
        move (str): The move in SAN notation to validate.
        optimal_move (str): The optimal move in SAN notation.
        legal_moves (List[str]): The legal moves in SAN notation.
        logs (dict): Dictionary to store logging metrics.
        success_reward (float): Reward for a correct move.
        weak_penalty_reward (float): Reward for a valid but non-optimal move.
        strong_penalty_reward (float): Penalty for an invalid move.
        
    Returns:
        Tuple containing (score, logs)
    """
    board = chess.Board(fen)
 
    # strong penalty for
    # 1. not following san format
    # 2. following san format but illegal move
    try:
        chess_move = board.parse_san(move)
    except ValueError:
        return strong_penalty_reward, logs
    if move not in legal_moves:
        return strong_penalty_reward, logs
    else:
        logs['legal_move'] = 1

    # weak penalty for not optimal move
    # rewarding for optimal move
    if move != optimal_move:
        return weak_penalty_reward, logs
    else:
        logs['optimal'] = 1
        return success_reward, logs

def compute_score(
    solution_str: str,
    ground_truth_dict: Dict[str, str],
    format_reward: float = 0.1,
    answer_reward: float = 2.0,
    weak_penalty_reward: float = 0.0,
    strong_penalty_reward: float = -2.0
) -> float:
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string.
        ground_truth_dict: Dictionary containing ground truth data.
        format_reward: Points awarded for correct format.
        answer_reward: Points awarded for a correct answer.
        
    Returns:
        Total score (sum of format and answer rewards)
    """

    # Parse ground truth data
    board_fen = ground_truth_dict['board_fen']
    next_move_san = ground_truth_dict['next_move_san']
    legal_moves_san = ground_truth_dict['legal_moves_san']

    # Initialize log metrics
    logs = {'format': 0, 'legal_move': 0, 'optimal': 0}

    # Extract model answer (pass logs to the function)
    answer_text, processed_str, logs = extract_solution(solution_str, logs)
    
    # Validate response structure
    format_correct, logs = validate_response_structure(processed_str, logs)
    format_score = format_reward if format_correct else strong_penalty_reward

    # Validate answer content
    answer_score = 0
    try:
        if format_correct and answer_text:
            answer_score, logs = validate_chess_move(
                fen=board_fen,
                move=answer_text,
                optimal_move=next_move_san,
                legal_moves=legal_moves_san,
                logs=logs,
                success_reward=answer_reward,
                weak_penalty_reward=weak_penalty_reward,
                strong_penalty_reward=strong_penalty_reward
            )
    except Exception as e:
        answer_score = strong_penalty_reward

    total_score = format_score + answer_score

    return total_score, logs
