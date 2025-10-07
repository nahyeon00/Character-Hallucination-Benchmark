#!/bin/bash

# 실행할 스크립트 리스트
scripts=(
  "context_test_vllm.sh"
  "fewshot_cot_test_vllm.sh"
  "reasoning_test_vllm.sh"
  "local_lang_test_vllm.sh"
)

# 순차적으로 실행
for script in "${scripts[@]}"; do
  echo "=== Running $script ==="
  bash "$script"
  if [ $? -ne 0 ]; then
    echo "Error occurred while running $script. Stopping execution."
    exit 1
  fi
  echo "=== Finished $script ==="
done

echo "All scripts executed successfully."
