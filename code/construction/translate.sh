#!/usr/bin/env bash
set -euo pipefail

# Array of target languages
languages=("Chinese" "Korean" "Spanish" "Indonesian")

# Locate this script’s directory so we can call the Python scripts reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
META_SCRIPT="$SCRIPT_DIR/translate_meta.py"
TEST_SCRIPT="$SCRIPT_DIR/translate_test_data.py"

for lang in "${languages[@]}"; do
  echo
  # echo "=== Translating meta data to $lang ==="
  # if ! python3 "$META_SCRIPT" --target_language "$lang"; then
  #   echo "✖ Error: translate_meta.py failed for $lang" >&2
  #   continue
  # fi
  # echo "✔ Meta data translation to $lang completed."

  echo "=== Translating test data to $lang ==="
  if ! python3 "$TEST_SCRIPT" --target_language "$lang"; then
    echo "✖ Error: translate_test_data.py failed for $lang" >&2
    continue
  fi
  echo "✔ Test data translation to $lang completed."
done

echo
echo "All translations finished."
