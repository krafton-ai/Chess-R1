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
run_script "/home/jovyan/chessLLM/scripts/chess_eval/qwen7b_base/eval_qwen7b_chess_comparison.sh"
run_script "/home/jovyan/chessLLM/scripts/chess_eval/qwen7b_base/eval_qwen7b_chess_deepmind_lichess.sh"
run_script "/home/jovyan/chessLLM/scripts/chess_eval/qwen7b_base/eval_qwen7b_chess_modeling_instruct.sh"

echo "All scripts executed successfully!"