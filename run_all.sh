#!/usr/bin/env sh
set -e

# Save the base directory
BASE_DIR=$(pwd)

# 1. Build all constructions
echo "[`date '+%Y-%m-%d %H:%M:%S'`] ▶ Running make_all_construction.sh..."
pushd code/construction > /dev/null
bash run_all_construction.sh
popd > /dev/null
echo "[`date '+%Y-%m-%d %H:%M:%S'`] ✔ make_all_construction.sh completed."

# 2. Run vLLM model per prompt evaluation
echo "[`date '+%Y-%m-%d %H:%M:%S'`] ▶ Running run_vllm_model_per_prompt.sh..."
pushd code/evaluation > /dev/null
bash run_vllm_model_per_prompt.sh
popd > /dev/null
echo "[`date '+%Y-%m-%d %H:%M:%S'`] ✔ run_vllm_model_per_prompt.sh completed."

echo "All tasks finished successfully."
