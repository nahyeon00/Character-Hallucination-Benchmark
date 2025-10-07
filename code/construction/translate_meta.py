import os
import json
import yaml
import time
from openai import OpenAI
from tqdm import tqdm
from typing import Any
import argparse


def load_config(config_file: str) -> dict:
    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def safe_translate(client: OpenAI, text: str, target_language: str, retries: int = 3, delay: float = 1.0) -> str:
    if not text or text.strip() == "":
        return text

    for attempt in range(retries):
        try:
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
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"[Error] Translation failed for: {text[:30]}... -> {e}")
                return text  # 실패 시 원문 반환

def translate_character_data(client: OpenAI, data: dict, target_language: str = "Korean") -> dict:
    for country in tqdm(data, desc="Processing countries"):
        for character in tqdm(data[country], desc=f"Processing {country}", leave=False):
            char_data = data[country][character]

            char_data['name'] = safe_translate(client, char_data.get('name', ""), target_language)
            char_data['profile'] = safe_translate(client, char_data.get('profile', ""), target_language)
            
            context = char_data.get('context', {})
            context['Nationality'] = safe_translate(client, context.get('Nationality', ""), target_language)
            context['Summary'] = safe_translate(client, context.get('Summary', ""), target_language)

    return data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../../data/source_data/meta_character.json")
    parser.add_argument("--target_language", type=str, default="Korean")
    args = parser.parse_args()

    config_path = "../../config.yaml"
    output_path = f"../../data/source_data/translated_{args.target_language}_meta_character.json"

    cfg = load_config(config_path)
    client = OpenAI(api_key=cfg["openai_key"])

    input_data = read_json(args.input_path)
    translated_data = translate_character_data(client, input_data, target_language=args.target_language)
    write_json(translated_data, output_path)

    print(f"\n✅ 번역 완료! 저장 위치: {output_path}")

if __name__ == "__main__":
    main()
