import re
import random
import chess
from typing import Dict, Tuple, Optional

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
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False
            logs['format'] = 0
            return validation_passed, logs

    # Verify tag order: <think> ... </think> ... <answer> ... </answer>
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
        logs['format'] = 0
    else:
        logs['format'] = 1
    
    return validation_passed, logs

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

def validate_chess_move(
    fen: str, 
    move: str, 
    optimal_move: str, 
    logs: dict,
    success_reward: float = 1.0,
    weak_penalty_reward: float = -1.0,
    strong_penalty_reward: float = -2.0,
) -> Tuple[float, dict]:
    """
    Validates a chess move for a given FEN string and assigns a score based on the validation rules.
    
    Args:
        fen (str): The FEN string representing the board state.
        move (str): The move in UCI notation to validate.
        optimal_move (str): The optimal move in UCI notation.
        logs (dict): Dictionary to store logging metrics.
        success_reward (float): Reward for a correct move.
        weak_penalty_reward (float): Reward for a valid but non-optimal move.
        strong_penalty_reward (float): Penalty for an invalid move.
        
    Returns:
        Tuple containing (score, logs)
    """
    # Check UCI notation format
    if not re.match(r'^[a-h][1-8][a-h][1-8](?:[qrbn])?$', move):
        #print(f"Move {move} does not follow UCI notation.")
        return strong_penalty_reward, logs  # Strong penalty
    else:
        logs['legal_uci'] = 1

    board = chess.Board(fen)
    try:
        chess_move = board.parse_uci(move)
    except ValueError:
        #print(f"Move {move} cannot be parsed, possibly due to moving a non-existent piece.")
        return strong_penalty_reward, logs  # Strong penalty

    legal_moves = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION_DICT[x.uci()])
    if chess_move not in legal_moves:
        print(f"Move {move} is not legal on the given board.")
        return strong_penalty_reward, logs  # Strong penalty
    else:
        logs['legal_move'] = 1

    if move == optimal_move:
        #print(f"Move {move} is optimal!")
        logs['optimal'] = 1
        return success_reward, logs  # Correct move case
    else:
        #print(f"Move {move} is legal but not optimal.")
        return weak_penalty_reward, logs  # Weak penalty

def compute_score(solution_str: str, 
                 ground_truth_dict: Dict[str, str],
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
    logs = {'format': 0, 'legal_uci': 0, 'legal_move': 0, 'optimal': 0}

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
                                    fen=fen_state,
                                    move=answer_text,
                                    optimal_move=solution_text,
                                    logs=logs,
                                    success_reward=answer_reward,
                                    weak_penalty_reward=weak_penalty_reward,
                                    strong_penalty_reward=strong_penalty_reward)
    except Exception as e:
        print("\n[Content Validation] Skipped due to format errors or missing answer:", e)
        answer_score = strong_penalty_reward

    total_score = format_score + answer_score
    do_print = random.randint(1, 64) == 1
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
