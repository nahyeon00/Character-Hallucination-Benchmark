import json
import os
import random
from tqdm import tqdm
from openai import OpenAI
import yaml

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

cfg = load_config("../../config.yaml")
client = OpenAI(api_key=cfg["openai_key"])

def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
# GPT-4를 이용해 보기 문장을 서술형으로 변환
def get_descriptive_sentence(prompt_template, country, item, question):

    prompt = prompt_template.format(
        item=item,
        question=question
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

# 전체 JSON 처리
def process_json(data, output_path, prompt_path):
    
    prompt_template = load_prompt_template(prompt_path)

    for entry in tqdm(data):
        question = entry["en_question"]
        
        descriptive_choices = {}
        for country, items in entry["country_choices"].items():
            item = items[0]
            descriptive = get_descriptive_sentence(prompt_template, country, item, question)
            descriptive_choices[country] = [descriptive]

        entry["formatted_choices"] = descriptive_choices

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def construct_mc_from_country(item, target_country):

    question = item['en_question']
    country_choices = item['country_choices']
    distractor_countries = [c for c in item['country_list'] if c != target_country]

    if target_country in country_choices:

        correct_choice = country_choices[target_country][0]

        # distractor (음식, 나라)
        distractors = [(country_choices[c][0], c) for c in distractor_countries if c in country_choices]
        if len(distractors) >= 3:

            random.shuffle(distractors)
            selected_distractors = distractors[:3]
            selected_distractors.append(("I can not answer that question.", "X"))

            all_choices = selected_distractors + [(correct_choice, target_country)]
            random.shuffle(all_choices)

            # (정답 텍스트, 정답 국가) 기준으로 위치 찾기
            correct_index = next(
                i for i, (text, country) in enumerate(all_choices)
                if text == correct_choice and country == target_country
            ) + 1

            entry = {
                "Question": question,
                "Answer": correct_choice,
                "True Label": correct_index
            }

            country_list_for_this_question = []

            for idx, (text, country) in enumerate(all_choices):
                key = ["one", "two", "three", "four", "five"][idx]
                entry[key] = text
                country_list_for_this_question.append(country)

            entry["country_list"] = country_list_for_this_question

            return entry
        
if __name__ == "__main__":
    
    input_json_path = '../../data/source_data/meta_cultural_qa.json'

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompt_path = "../../prompt/cultural_tolong.txt"
    prompt_template = load_prompt_template(prompt_path)

    country_list = list(data.keys())
    raw_result_data = {}
    for country in country_list:
        country = country.strip()
        raw_result_data[country] = []

    for ctr in tqdm(data):
        for entry in data[ctr]:
            question = entry["en_question"]

            descriptive_choices = {}
            for country, items in entry["country_choices"].items():
                item = items[0]
                descriptive = get_descriptive_sentence(prompt_template, country, item, question)
                descriptive_choices[country] = [descriptive]
                
                entry["formatted_choices"] = descriptive_choices

            raw_result_data[ctr].append(entry)


    raw_output_json_path = f"../../data/source_data/cultural_choices_descriptive.json"

    with open(raw_output_json_path, 'w', encoding='utf-8') as f:
        json.dump(raw_result_data, f, ensure_ascii=False, indent=2)
