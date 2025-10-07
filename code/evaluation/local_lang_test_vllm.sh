#!/usr/bin/env sh
set -e

LOG_DIR="./log"
mkdir -p "$LOG_DIR"

# LANGUAGE_LIST=("Korean" "Chinese" "Spanish" "Indonesian")
LANGUAGE_LIST=("Spanish")

# 실행할 모델 목록
MODELS=(
  # "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-Nemo-Instruct-2407"
  # "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  "Qwen/Qwen3-8B"
  # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
DEVICE="1,2"

QUESTION_TYPES="cross temporal cultural fact"

CONTEXT_COMBINATIONS=(
"no_context"
# "birth"
# "Nationality"
# "Summary"
# "birth Nationality"
# "Nationality Summary"
# "birth Summary"
# "birth Nationality Summary"
)


for LANGUAGE in "${LANGUAGE_LIST[@]}"; do 

    INPUT_DIR="../../data/test_data_check_fin_${LANGUAGE}_translated"


    CHAR_DIR="../../data/source_data/translated_${LANGUAGE}_meta_character.json"

    # 프롬프트 템플릿 경로 리스트
    PROMPT_TEMPLATE_PATHS=(
        "../../prompt/mc_eval_template_0-shot_translate_${LANGUAGE}.txt"
    )

  for MODEL in "${MODELS[@]}"; do
    for PROMPT_PATH in "${PROMPT_TEMPLATE_PATHS[@]}"; do
      for CONTEXT_TYPES in "${CONTEXT_COMBINATIONS[@]}"; do
        CTX_CLEAN=$(echo "$CONTEXT_TYPES" | tr ' ' '_')  # 예: "birth Nationality" → "birth_Nationality"
        MODEL_NAME_CLEAN=$(echo "$MODEL" | tr '/ ' '__')

        for QTYPE in $QUESTION_TYPES; do
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
    done
  done

  echo "모든 작업 완료."
done