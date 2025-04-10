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

def compute_score(
    solution_str: str,
    ground_truth_dict: Dict[str, str],
    answer_reward: float = 1.0,
) -> float:
    """
    Computes comprehensive score for model response.
    """

    if 'Given a FEN string, draw the current chessboard position and enclose it within <answer>...</answer> tags.' in solution_str:
        cur_data_type = 'draw_chessboard'
    elif 'Generate all legal moves in SAN format for the given FEN board state and enclose it within <answer>...</answer> tags.' in solution_str:
        cur_data_type = 'legal_moves'
    elif 'Generate the resulting FEN position given the following SAN moves, starting from the initial FEN.' in solution_str:
        cur_data_type = 'consecutive_SAN_FEN'
    else:
        raise ValueError('Unknown instruction')

    # Parse ground truth data
    answer = ground_truth_dict['answer']

    # Initialize log metrics
    logs = {'format': 0, f'{cur_data_type}/accuracy': 0, }

    # Extract model answer (pass logs to the function)
    answer_text, processed_str, logs = extract_solution(solution_str, logs)
    
    try:
        if answer_text:
            correct = answer_text == answer
            if correct:
                logs[f'{cur_data_type}/accuracy'] = 1
        else:
            correct = 0.0
    except Exception as e:
        correct = 0.0
        print("\n[Content Validation] Skipped due to format errors or missing answer:", e)                

    return correct, logs
