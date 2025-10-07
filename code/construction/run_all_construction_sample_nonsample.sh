#!/usr/bin/env bash
set -euo pipefail

# 입력 파일 디렉터리
INPUT_DIR="../../data/source_data/meta_character.json"

# 순회할 버전 리스트
VERSIONS=("version3")

# make_fact_* 에 사용할 question_type
MAKE_TYPES=("cross" "temporal" "fact")

# shuffle_fact_* 에 사용할 question_type
SHUFFLE_TYPES=("cross" "temporal" "fact")

for VER in "${VERSIONS[@]}"; do
  echo "=== prompt_version = $VER ==="

  # 1) make_fact_cross_temporal_choices.py 실행 (병렬)
  for QT in "${MAKE_TYPES[@]}"; do
    {
      echo "-> make_fact_cross_temporal_choices.py --question_type $QT --prompt_version $VER"
      python make_fact_cross_temporal_choices.py \
        --input_dir "$INPUT_DIR" \
        --question_type "$QT" \
        --prompt_version "$VER"
    } &
  done

  # 2) shuffle_fact_cross_temporal.py 실행 (sample_choice=False, 병렬)
  for QT in "${SHUFFLE_TYPES[@]}"; do
    {
      echo "-> shuffle_fact_cross_temporal.py --question_type $QT --prompt_version $VER"
      python shuffle_fact_cross_temporal.py \
        --question_type "$QT" \
        --prompt_version "$VER" \
        --sample_choice False
    } &
  done

  # 백그라운드로 띄운 모든 작업이 끝날 때까지 대기
  wait

  echo
done
