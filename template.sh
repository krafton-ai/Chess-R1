#!/usr/bin/env bash

SCRIPT="./examples/data_preprocess/deepmind_lichess_quiz.py"

# List of all template types defined in make_prefix()
TEMPLATES=(
  "qwen_instruct_fen"
  "qwen_instruct_fen_legal"
  "qwen_instruct_fen_legal_rule"
  "qwen_instruct_fen_legal_rule_table"
  "qwen_instruct_fen_legal_rule_table_pgn-full"
)

for template in "${TEMPLATES[@]}"; do
  echo "=============================================="
  echo "Running template: $template"
  echo "=============================================="
  python "$SCRIPT" \
    --template_type "$template" 
done

echo "All templates processed!"