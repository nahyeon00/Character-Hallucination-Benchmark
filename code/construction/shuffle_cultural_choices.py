import json
import os
import random
from tqdm import tqdm
import argparse


def construct_mc_from_country(data, target_country):
    result_entries = []

    for item in data:
        question = item['en_question']
        formatted = item.get('formatted_choices', {})
        distractor_countries = [c for c in item['country_list'] if c != target_country]

        if target_country not in formatted:
            continue


        correct_choice = formatted[target_country]


        distractors = [
            (formatted[c], c)
            for c in distractor_countries
            if c in formatted
        ]
        if len(distractors) < 3:
            continue

        random.shuffle(distractors)
        selected_distractors = distractors[:3]
        selected_distractors.append((["I can not answer that question."], "X"))

        all_choices = selected_distractors + [(correct_choice, target_country)]
        random.shuffle(all_choices)

        correct_index = next(
            i for i, (text, country) in enumerate(all_choices)
            if text == correct_choice and country == target_country
        ) + 1

        entry = {
            "Question": question,
            "Answer": correct_choice[0],
            "True Label": correct_index
        }

        country_list_for_this_question = []
        for idx, (text, country) in enumerate(all_choices):
            key = ["one", "two", "three", "four", "five"][idx]
            entry[key] = text[0]
            country_list_for_this_question.append(country)

        entry["country_list"] = country_list_for_this_question
        result_entries.append(entry)

    return result_entries

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, default='../../data/source_data/cultural_choices_descriptive.json')
    parser.add_argument("--output_folder_name", type=str, default="test_data")
    args = parser.parse_args()

    random.seed(42)


    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    country_list = list(data.keys())

    result_data = {}
    for country in country_list:
        country = country.strip()
        result_data[country] = []

    for country in tqdm(data):
        shuffled = construct_mc_from_country(data=data[country], target_country=country)
        result_data[country] = shuffled


    output_json_path = f"../../data/{args.output_folder_name}/cultural_mc.json"

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
