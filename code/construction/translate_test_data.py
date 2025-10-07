import os
import json
import yaml
from openai import OpenAI
from tqdm import tqdm
import argparse

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

cfg = load_config("../../config.yaml")

# OpenAI API client 초기화
client = OpenAI(api_key=cfg["openai_key"])

# 번역 함수 (target_language 매개변수 추가)
def translate(text, target_language="Korean"):
    if not text or text.strip() == "":
        return text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
            "content": (
                f"You are a professional translation engine. "
                f"Translate the following English text strictly into {target_language}. "
                f"Do not add, omit, rephrase, summarize, continue, or interpret the text. "
                f"Preserve the meaning and style exactly. "
                f"Your response must be the translation only — nothing else."
        )},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# 대상 텍스트 키들
TEXT_KEYS = ["Question", "Answer", "one", "two", "three", "four", "five"]

# JSON 변환 함수
def translate_json_file(input_path, output_path, target_language="Korean"):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def translate_entry(entry):
        for key in TEXT_KEYS:
            if key in entry:
                entry[key] = translate(entry[key], target_language)
        return entry

    # 구조가 다를 수 있으므로 구조별 처리
    if isinstance(data, list):
        translated_data = [translate_entry(entry) for entry in tqdm(data)]
    elif isinstance(data, dict):
        # cross_mc / cultural_mc 스타일인지 확인
        sample_value = next(iter(data.values()))
        if isinstance(sample_value, dict):
            translated_data = {}
            for outer_key, characters in data.items():
                translated_data[outer_key] = {}
                for character_key, items in tqdm(characters.items(), desc=f"{outer_key}"):
                    translated_data[outer_key][character_key] = [translate_entry(item) for item in items]
        else:
            translated_data = {}
            for outer_key, outer_value in data.items():
                translated_data[outer_key] = [translate_entry(item) for item in tqdm(outer_value, desc=f"{outer_key}")]
    else:
        raise ValueError("Unsupported JSON structure")

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='../../data/test_data_fin_en')
    parser.add_argument("--target_language", type=str, default="Korean")
    args = parser.parse_args()

    # ① output_folder 자체를 생성
    output_folder = f'../../data/{args.input_path}_{args.target_language}_translated'
    os.makedirs(output_folder, exist_ok=True)

    file_list = [
        'cross_mc.json',
        'cultural_mc.json',
        'fact_mc.json',
        'temporal_mc.json'
    ]

    for file_name in file_list:
        input_path = os.path.join(args.input_path, file_name)
        output_path = os.path.join(output_folder, file_name)
        print(f"Processing {file_name} to {args.target_language}...")
        translate_json_file(input_path, output_path, target_language=args.target_language)
        print(f"Saved translated file to {output_path}\n")

    print(f"\n✅ 번역 완료! 저장 위치: {output_folder}")

if __name__ == "__main__":
    main()