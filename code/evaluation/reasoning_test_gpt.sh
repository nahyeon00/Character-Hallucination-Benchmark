#!/usr/bin/env sh
set -e

LOG_DIR="./log"
mkdir -p "$LOG_DIR"

INPUT_DIR="../../data/test_data_check_fin"
CHAR_DIR="../../data/source_data/meta_character.json"

MODELS=(
  "o1"
#   "gpt-4o"
#   "gpt-3.5-turbo-0125"
)

DEVICE="0,1"

QUESTION_TYPES="cross temporal cultural fact"

CONTEXT_COMBINATIONS=(
  # "no_context"
  # "birth"
  # "Nationality"
  # "Summary"
  # "birth Nationality"
  # "Nationality Summary"
  # "birth Summary"
  "birth Nationality Summary"
)

# 프롬프트 템플릿 경로 리스트
PROMPT_TEMPLATE_PATHS=(
  "../../prompt/mc_eval_template_0-shot.txt"
#   "../../prompt/mc_eval_template_1-shot.txt"
#   "../../prompt/mc_eval_template_2-shot.txt"
#   "../../prompt/mc_eval_template_3-shot.txt"
  "../../prompt/mc_eval_template_3-shot_div.txt"
  "../../prompt/mc_eval_template_0-shot_cot.txt"
#   "../../prompt/mc_eval_template_1-shot_cot.txt"
#   "../../prompt/mc_eval_template_2-shot_cot.txt"
#   "../../prompt/mc_eval_template_3-shot_cot.txt"
  "../../prompt/mc_eval_template_3-shot_div_cot.txt"
)

for MODEL in "${MODELS[@]}"; do
  echo "=== Starting MODEL=$MODEL ==="

  for PROMPT_PATH in "${PROMPT_TEMPLATE_PATHS[@]}"; do
    (
      for CONTEXT_TYPES in "${CONTEXT_COMBINATIONS[@]}"; do
        for QTYPE in $QUESTION_TYPES; do
          CTX_CLEAN=$(echo "$CONTEXT_TYPES" | tr ' ' '_')
          MODEL_NAME_CLEAN=$(echo "$MODEL" | tr '/ ' '__')
          PROMPT_NAME=$(basename "$PROMPT_PATH" .txt)
          TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
          LOG_FILE_NAME="${MODEL_NAME_CLEAN}_${CTX_CLEAN}_${QTYPE}_${PROMPT_NAME}.log"

          echo "[${TIMESTAMP}] Running model=${MODEL}, question_type=${QTYPE}, context_types=\"${CONTEXT_TYPES}\", prompt_template=${PROMPT_PATH}..."

          if python script.py \
               --model_name "${MODEL}" \
               --input_dir_path "${INPUT_DIR}" \
               --meta_char_dir "${CHAR_DIR}" \
               --question_type "${QTYPE}" \
               --device_index "${DEVICE}" \
               --context_types ${CONTEXT_TYPES} \
               --prompt_template_path "${PROMPT_PATH}" \
             2>&1 | tee "${LOG_DIR}/${LOG_FILE_NAME}"; then
            echo "  → 성공: 로그 → ${LOG_DIR}/${LOG_FILE_NAME}"
          else
            echo "  **실패**: model=${MODEL}, question_type=${QTYPE}, context_types=\"${CONTEXT_TYPES}\", prompt_template=${PROMPT_PATH} 실행 중 에러."
            echo "         로그 → ${LOG_DIR}/${LOG_FILE_NAME}"
          fi

          echo
        done
      done
    ) &
  done

  wait
  echo "=== Finished MODEL=$MODEL ==="
  echo
done

echo "모든 작업 완료."
