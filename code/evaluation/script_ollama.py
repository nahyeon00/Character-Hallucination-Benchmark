import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import ollama
import torch
import random
import yaml
import re
import time
import warnings
import logging
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

def compute_accuracy_per_file(input_data):

    y_true = []
    y_pred = []

    for item in input_data:
        true_label = str(item.get("True Label")).strip()
        model_answer = str(item.get("model_answer_number")).strip()
        y_true.append(true_label)
        y_pred.append(model_answer)

    acc = accuracy_score(y_true, y_pred)
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    total = len(y_true)
    summary = {
        "accuracy": round(acc * 100, 2),
        "correct": correct,
        "total": total
    }

    return summary

def make_table(folder_name):
    # JSON 파일 이름에서 특정 키워드 추출
    keywords = []
    pattern = re.compile(r'(cultural|cross|fact|temporal)')

    data_dict = {}
    for json_file in folder_name.glob('*.json'):
        match = pattern.search(json_file.name)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_dict[match.group(1)] = data

    character_info = pd.read_json('../../data/source_data/meta_character.json')

    character_info = character_info.reset_index().melt(
        id_vars='index',
        value_vars=['china', 'en', 'korea', 'mexico', 'spain'],
        var_name='country',
        value_name='info'
    ).dropna(subset=['info']).rename(columns={'index': 'character'}).reset_index(drop=True)

    character_info_expanded = pd.concat(
        [character_info.drop(columns=['info']), character_info['info'].apply(pd.Series)],
        axis=1
    )

    result_dict = {}

    for question_type in data_dict:
        result_dict[question_type] = {}

        for country in data_dict[question_type]:
            result_dict[question_type][country] = {}
            for character in data_dict[question_type][country]:
                result_dict[question_type][country][character] = compute_accuracy_per_file(data_dict[question_type][country][character])

    # 정확도 계산 (기존 코드와 동일)
    average_accuracies = {}
    for q_type, countries in result_dict.items():
        average_accuracies[q_type] = {}
        for country, characters in countries.items():
            accuracies = [char_data['accuracy'] for char_data in characters.values()]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            average_accuracies[q_type][country] = round(avg_accuracy, 2)

    df = pd.DataFrame(average_accuracies).T
    df['Average'] = df.mean(axis=1).round(2)
    df.loc['Average'] = df.mean(axis=0).round(2)

    result_df = pd.DataFrame.from_dict(
        {
            (q_type, country): characters
            for q_type, countries in result_dict.items()
            for country, characters in countries.items()
        },
        orient='index'
    )

    result_df.index.names = ['Question Type', 'Country']

    result_df_reset = result_df.reset_index().melt(
        id_vars=['Question Type', 'Country'],
        var_name='character_name',
        value_name='accuracy_info'
    ).dropna(subset=['accuracy_info'])


    df = result_df_reset.copy()
    df[['accuracy', 'total']] = df['accuracy_info'].apply(pd.Series)[['accuracy', 'total']]


    agg = (
        df
        .groupby(['character_name', 'Question Type'])
        .agg(accuracy=('accuracy', 'mean'),
            total   =('total',    'sum'))
        .unstack('Question Type')   # Question Type을 컬럼으로 펼치기
    )

    new_cols = []
    for metric, qtype in agg.columns:
        if metric == 'accuracy':
            new_cols.append(f"{qtype}_accuracy")
        else:  # total
            new_cols.append(f"{qtype}_num")
    agg.columns = new_cols

    result = agg.reset_index()

    cols_to_check = [col for col in result.columns
                    if ('cross' not in col) and ('temporal' not in col)]

    filtered_result = result.dropna(subset=cols_to_check)
    info_cols = ['character', 'country', 'history', 'time']


    merged_df = (
        filtered_result
        .merge(
            character_info_expanded[info_cols],
            left_on='character_name',
            right_on='character',
            how='left'
        )
        .drop(columns='character')  # 중복된 key 컬럼 제거
    )

    cols_to_nan = [col for col in merged_df.columns if 'cross' in col or 'temporal' in col]
    merged_df.loc[merged_df['time'] == 'present', cols_to_nan] = np.nan

    groupings = {
        'Country': ['country'],
        'History': ['history'],
        'Time': ['time'],
        'History & Time': ['history', 'time']
    }
    markdown_lines = []
    for name, cols in groupings.items():
        # ----- Accuracy Table -----
        acc = merged_df.groupby(cols)[[
            'cross_accuracy', 'cultural_accuracy',
            'fact_accuracy', 'temporal_accuracy'
        ]].mean().round(2)
        acc['row_avg'] = acc.mean(axis=1).round(2)
        acc.loc['col_avg'] = acc.mean().round(2)

        markdown_lines.append(f"### Accuracy by {name}\n")
        markdown_lines.append(acc.to_markdown())
        markdown_lines.append("\n")

        # ----- Num Table -----
        nums = merged_df.groupby(cols)[[
            'cross_num', 'cultural_num',
            'fact_num', 'temporal_num'
        ]].sum()
        nums['row_sum'] = nums.sum(axis=1)
        nums.loc['col_sum'] = nums.sum()

        markdown_lines.append(f"### Num by {name}\n")
        markdown_lines.append(nums.to_markdown())
        markdown_lines.append("\n")

    # result.md 파일로 쓰기
    output_path = folder_name / 'result.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))

    print(f"✅ 마크다운 파일을 생성했습니다: {output_path}")



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def open_json(file_dir):
    with open(file_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def extract_model_answer_number(model_answer):
    match = re.search(r'\b([1-5])\b', model_answer)
    if match:
        return int(match.group(1))
    return None

def run_mc_evaluation(mc_list, model_name, name, context, template, is_gpt=False,
                      client=None, llm=None, sampling_params=None,
                      batch_size=64, temperature=0.0, seed=42):
    set_seed(seed)
    result_data = []


    if is_gpt:
        for row in mc_list:
            prompt = template.format(
                character=name, profile=context,
                Question=row['Question'],
                answer1=row['one'], answer2=row['two'],
                answer3=row['three'], answer4=row['four'], answer5=row['five']
            )
            messages = [
                {"role": "system", "content": f"I want you to act like {name}"},
                {"role": "user", "content": prompt},
            ]

            with open('./prompt_text.txt', "w", encoding="utf-8") as f:
                f.write("----- PROMPT -----\n")
                f.write(prompt + "\n\n")
                f.write("----- MESSAGES -----\n")
                # JSON 형태로 보기 좋게 저장하고 싶으면 json.dumps 사용
                f.write(json.dumps(messages, ensure_ascii=False, indent=2))

            outputs = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=512,
                n=1,
                top_p=0.95,
            )
            response = outputs.choices[0].message.content.strip()
            model_answer = response.split("\n")[-1]
            result_data.append({
                "Question": row['Question'],
                "True Label": row['True Label'],
                "one": row['one'], "two": row['two'], "three": row['three'],
                "four": row['four'], "five": row['five'],
                "model_result": response,
                "model_answer": model_answer,
                "model_answer_number": extract_model_answer_number(model_answer)
            })
    else:
        for row in mc_list:
            prompt_text = template.format(
                character=name, profile=context,
                Question=row["Question"],
                answer1=row["one"], answer2=row["two"],
                answer3=row["three"], answer4=row["four"], answer5=row["five"]
            )
            resp = ollama.generate(
                model=model_name,
                prompt=prompt_text,
                options={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "num_predict": 64,
                    "seed": seed,
                },
            )
            generated_text = resp.get("response", "").strip()
            model_answer = generated_text
            result_data.append({
                "Question": row["Question"],
                "True Label": row["True Label"],
                "one": row["one"], "two": row["two"], "three": row["three"],
                "four": row["four"], "five": row["five"],
                "model_result": generated_text,
                "model_answer": model_answer,
                "model_answer_number": extract_model_answer_number(model_answer),
            })
    return result_data

if __name__ == "__main__":
    start_time = time.time()

    cfg = load_config("../../config.yaml")

    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument("--question_type", type=str, default="cross")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--context_types", nargs="*", default=["birth", "Nationality", "Summary"])
    parser.add_argument("--meta_char_dir", type=str, default="../../data/source_data/meta_character.json")
    parser.add_argument("--input_dir_path", type=str, default="../../data/test_data")
    parser.add_argument("--device_index", type=str, help="GPU device indices, comma-separated (예: 0,1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_template_path", type=str, default="../../prompt/mc_eval_template_0_shot.txt")
    args = parser.parse_args()

    character_info = open_json(file_dir=args.meta_char_dir)
    data = open_json(file_dir=f"{args.input_dir_path}/{args.question_type}_mc.json")

    result_dic = {}

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device_indices = args.device_index.split(",")
    tensor_parallel_size = len(device_indices)

    filename = f"{args.question_type}_evaluation_result.json"

    prompt_template_name = args.prompt_template_path.split("mc_eval_template_")[-1]
    prompt_template_name = prompt_template_name.split(".txt")[0]
    output_folder_name = args.model_name.split("/")[-1] if "gpt" not in args.model_name else args.model_name
    if args.input_dir_path.split("/")[-1] != "test_data":
        output_folder_name = f"{output_folder_name}_{args.input_dir_path.split('/')[-1]}_{prompt_template_name}"


    output_dir = f"../../data/prediction_data/{output_folder_name}/{str(args.context_types)}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Model & template loading (only once)
    is_gpt = "gpt" in args.model_name.lower()
    client = OpenAI(api_key=cfg["openai_key"]) if is_gpt else None
    # template_path = "../../prompt/mc_eval_template_gpt.txt" if is_gpt else "../../prompt/mc_eval_template_llama.txt"
    template_path = args.prompt_template_path
    template = load_template(template_path)
    sampling_params = None  # ollama does not require SamplingParams
    llm = None  # placeholder for compatibility

    for country in tqdm(character_info):
        if args.question_type in ["cross", "fact", "cultural"] and country not in data:
            logger.warning(f"No MC data for country '{country}', skipping.")  # 선택
            continue
        result_dic[country] = {}
        for character in character_info[country]:
            if args.question_type in ["cross", "fact"]:
                if character not in data[country]:
                    continue
                mc_list_data = data[country][character]
            elif args.question_type == "cultural":
                mc_list_data = data[country]
            elif args.question_type == "temporal":
                mc_list_data = data

            char_name = character
            char_profile = character_info[country][character]['profile']
            if args.context_types[0] == "no_context":
                raw_context = char_profile
            else:
                raw_context = {
                    label: character_info[country][character]['context'][label]
                    for label in args.context_types
                }
            if isinstance(raw_context, dict):
                context_lines = [f'"{k}": "{v}"' for k, v in raw_context.items()]
                char_context = "\n".join(context_lines)
            else:
                char_context = f'"""\n{raw_context}\n"""'
        
            mc_return_list = run_mc_evaluation(
                mc_list=mc_list_data,
                model_name=args.model_name,
                name=char_name,
                context=char_context,
                template=template,
                is_gpt=is_gpt,
                client=client,
                llm=llm,
                sampling_params=sampling_params,
                batch_size=32,
                temperature=args.temperature,
                seed=args.seed
            )

            result_dic[country][character] = mc_return_list
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dic, f, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    output_folder_dir = f"../../data/prediction_data/{output_folder_name}"
    folder_paths = [folder for folder in Path(output_folder_dir).iterdir() if folder.is_dir()]

    for base_path in folder_paths:
        make_table(base_path)
