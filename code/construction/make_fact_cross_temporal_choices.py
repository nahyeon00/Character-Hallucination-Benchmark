import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import yaml

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def load_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def query_gpt(prompt, model="gpt-4o", api_key=None):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a plausible incorrect answer generator for multiple choice question datasets."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512,
        top_p=0.95,
    )
    return response.choices[0].message.content.strip()

def generate_step1(name, profile, question, answer, api_key, type_name, use_profile):
    template_path = f"../../prompt/neg1_{'profile' if use_profile else 'nonprofile'}.txt"
    template = load_template(template_path)

    if use_profile:
        prompt = template.format(
            profile=profile,
            Question=question,
            Answer=answer
        )
    else:
        prompt = template.format(
            Question=question,
            Answer=answer
        )

    raw_output = query_gpt(prompt, api_key=api_key)
    answers = [line.split(": ", 1)[1] for line in raw_output.split("\n") if line.startswith("Incorrect Answer")]
    answers += [""] * (4 - len(answers))

    result_data = {
        "Question": question,
        "Answer": answer,
        "Incorrect Answer 1": answers[0],
        "Incorrect Answer 2": answers[1],
        "Incorrect Answer 3": answers[2],
        "Incorrect Answer 4": answers[3],
    }
    
    return result_data

def generate_step2(name, profile, question, answer, api_key, type_name, use_profile, previous_result, version):
    if version == "original":
        template_path = f"../../prompt/neg2_{'profile' if use_profile else 'nonprofile'}.txt"
    elif version == "version1":
        template_path = f"../../prompt/neg2_{'profile' if use_profile else 'nonprofile'}_version1.txt"
    elif version == "version2":
        template_path = f"../../prompt/neg2_{'profile' if use_profile else 'nonprofile'}_version2.txt"
    elif version == "version3":
        template_path = f"../../prompt/neg2_{'profile' if use_profile else 'nonprofile'}_version3.txt"
    template = load_template(template_path)
    print(f"version : {version}")
    print(template)

    if use_profile:
        prompt = template.format(
            profile=profile,
            Question=question,
            Answer=answer,
            Incorrect1=previous_result["Incorrect Answer 1"],
            Incorrect2=previous_result["Incorrect Answer 2"],
            Incorrect3=previous_result["Incorrect Answer 3"],
            Incorrect4=previous_result["Incorrect Answer 4"]
        )
    else:
        prompt = template.format(
            Question=question,
            Answer=answer,
            Incorrect1=previous_result["Incorrect Answer 1"],
            Incorrect2=previous_result["Incorrect Answer 2"],
            Incorrect3=previous_result["Incorrect Answer 3"],
            Incorrect4=previous_result["Incorrect Answer 4"]
        )

    raw_output = query_gpt(prompt, api_key=api_key)
    answers = [line.split(": ", 1)[1] for line in raw_output.split("\n") if line.startswith("Incorrect Answer")]
    answers += [""] * (4 - len(answers))

    result_data = {
            "Question": question,
            "Answer": answer,
            "Incorrect Answer 1": previous_result["Incorrect Answer 1"],
            "Incorrect Answer 2": previous_result["Incorrect Answer 2"],
            "Incorrect Answer 3": previous_result["Incorrect Answer 3"],
            "Incorrect Answer 4": previous_result["Incorrect Answer 4"],
            "Incorrect Answer 5": answers[0],
            "Incorrect Answer 6": answers[1],
            "Incorrect Answer 7": answers[2],
            "Incorrect Answer 8": answers[3],
        }

    if type_name.strip().lower() == "fact":
        result_data["Incorrect Answer 9"] = "I can not answer that question."

    return result_data

def generate_step3(name, profile, question, answer, api_key, type_name, use_profile, previous_result):
    template_path = f"../../prompt/neg3_{'profile' if use_profile else 'nonprofile'}.txt"
    template = load_template(template_path)


    if use_profile:
        prompt = template.format(
            profile=profile,
            Question=question,
            Answer=answer,
        )
    else:
        prompt = template.format(
            Question=question,
            Answer=answer,
        )

    raw_output = query_gpt(prompt, api_key=api_key)
    answers = [line.split(": ", 1)[1] for line in raw_output.split("\n") if line.startswith("Incorrect Answer")]
    answers += [""] * (2 - len(answers))

    result_data = {
            "Question": question,
            "Answer": answer,
            "Incorrect Answer 1": previous_result["Incorrect Answer 1"],
            "Incorrect Answer 2": previous_result["Incorrect Answer 2"],
            "Incorrect Answer 3": previous_result["Incorrect Answer 3"],
            "Incorrect Answer 4": previous_result["Incorrect Answer 4"],
            "Incorrect Answer 5": previous_result["Incorrect Answer 5"],
            "Incorrect Answer 6": previous_result["Incorrect Answer 6"],
            "Incorrect Answer 7": previous_result["Incorrect Answer 7"],
            "Incorrect Answer 8": previous_result["Incorrect Answer 8"],
            "Incorrect Answer 9": answers[0],
            "Incorrect Answer 10": answers[1],
        }

    return result_data


if __name__ == "__main__":
    cfg = load_config("../../config.yaml")

    parser = argparse.ArgumentParser(description="Generate MC dataset using OpenAI GPT API.")
    parser.add_argument("--input_dir", type=str, default="../../data/source_data/meta_character.json",help="Path to the input Excel file.")
    parser.add_argument("--output_dir", type=str, default="../../data/source_data", help="Directory to save output files.")
    parser.add_argument("--question_type", type=str, required=True, help="Prefix for file naming (e.g., temporal, cultural, cross etc.)")
    parser.add_argument("--prompt_version", type=str, default="original")

    args = parser.parse_args()

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "sample" in args.input_dir:
        if args.question_type == "cross":
            df = pd.read_csv('../../data/source_data/cross_universe_qa_sample.csv')
            country_list = list(set(df['country']))
        elif args.question_type == "fact":
            df = pd.read_csv('../../data/source_data/factural_qa_sample.csv')
            country_list = list(set(df['country']))
        elif args.question_type == "temporal":
            df = pd.read_csv('../../data/source_data/temporal_qa.csv')
            df = df[:10]
            country_list = None

    else:
        if args.question_type == "cross":
            df = pd.read_csv('../../data/source_data/cross_universe_qa.csv')
            country_list = list(set(df['country']))
        elif args.question_type == "fact":
            df = pd.read_csv('../../data/source_data/factural_qa.csv')
            country_list = list(set(df['country']))
        elif args.question_type == "temporal":
            df = pd.read_csv('../../data/source_data/temporal_qa.csv')
            country_list = None

    if args.question_type=="temporal":
        use_profile = False
    else:
        use_profile = True

    if country_list != None:
        result_data = {}
        for country in country_list:
            country = country.strip()
            result_data[country] = {}
    else:
        result_data = []

    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
        if use_profile == True:
            name = row['character']
            profile = row['profile']
            country = row['country'].strip()
        else:
            name = []
            profile = []
            country = []
        question = row['question']
        answer = row['answer']

        step1_result = generate_step1(name=name, profile=profile, question=question, answer=answer, 
                                api_key=cfg["openai_key"], type_name=args.question_type, use_profile=use_profile)
        
        step2_result = generate_step2(name=name, profile=profile, question=question, answer=answer, 
                                api_key=cfg["openai_key"], type_name=args.question_type, use_profile=use_profile, previous_result = step1_result, version=args.prompt_version)
        
        final_result = step2_result

        if args.question_type != "fact":
            final_result = generate_step3(name=name, profile=profile, question=question, answer=answer, 
                                api_key=cfg["openai_key"], type_name=args.question_type, use_profile=use_profile, previous_result = step2_result)
            
        if args.question_type == "temporal":
            result_data.append(final_result)
        elif args.question_type in ["fact", "cross"]:
            if name not in result_data[country]:
                result_data[country][name] = []
            result_data[country][name].append(final_result)

    if "sample" in args.input_dir:
        filename = f"{args.question_type}_choices_sample_{args.prompt_version}.json"
    else:
        filename = f"{args.question_type}_choices_{args.prompt_version}.json"
    output_path = os.path.join(args.output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)