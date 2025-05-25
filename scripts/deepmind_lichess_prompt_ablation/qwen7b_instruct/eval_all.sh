#!/bin/bash

# Function to run a script and exit if it fails
run_script() {
  script_name="$1"
  echo "Running ${script_name}..."
  "${script_name}"
  if [ $? -ne 0 ]; then
    echo "Error: ${script_name} failed to execute correctly."
    exit 1
  fi
}

# Run each script consecutively
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen7b_instruct/eval_qwen7b_chess_fen.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen7b_instruct/eval_qwen7b_chess_fen_legal.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen7b_instruct/eval_qwen7b_chess_fen_legal_rule.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen7b_instruct/eval_qwen7b_chess_fen_legal_rule_table.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen7b_instruct/eval_qwen7b_chess_fen_legal_rule_table_pgn-full.sh"

run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen3b_instruct/eval_qwen3b_chess_fen.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen3b_instruct/eval_qwen3b_chess_fen_legal.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen3b_instruct/eval_qwen3b_chess_fen_legal_rule.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen3b_instruct/eval_qwen3b_chess_fen_legal_rule_table.sh"
run_script "/home/jovyan/chessLLM/scripts/deepmind_lichess_prompt_ablation/qwen3b_instruct/eval_qwen3b_chess_fen_legal_rule_table_pgn-full.sh"

echo "All scripts executed successfully!"