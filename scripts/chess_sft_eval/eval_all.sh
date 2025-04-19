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
run_script "/home/jovyan/chessLLM/scripts/chess_sft_eval/eval_chesssftqwen7b_chess_best_move.sh"
run_script "/home/jovyan/chessLLM/scripts/chess_sft_eval/eval_chesssftqwen7b_chess_comparison.sh"
run_script "/home/jovyan/chessLLM/scripts/chess_sft_eval/eval_chesssftqwen7b_chess_modeling_instruct.sh"

echo "All scripts executed successfully!"