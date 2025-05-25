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
run_script "/home/jovyan/chessLLM/scripts/ICLR_season2/Qwen25_7B_Base_chesspretrain3of5_reastemp_fen_legal_rule.sh"
run_script "/home/jovyan/chessLLM/scripts/ICLR_season2/Llama31_8B_Base_nochesspretrain_reastemp_fen_legal_rule.sh"

echo "All scripts executed successfully!"