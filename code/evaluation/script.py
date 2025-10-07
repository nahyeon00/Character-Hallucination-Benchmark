import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
import torch
import random
import yaml
import re
import time
import warnings
import logging
import numpy as np
from pathlib import Path
from time import perf_counter

from sklearn.metrics import accuracy_score


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def safe_div(num, den):
    """Vectorized safe division: returns NaN where denominator is 0.
    Works on numpy arrays or pandas Series/DataFrame .values.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan)
    np.divide(num, den, out=out, where=den != 0)
    return out


def compute_accuracy_per_file(input_data):
    """Compute accuracy, correct, total for a single (character × qtype × country) list.
    input_data: list of dicts with keys ["True Label", "model_answer_number"].
    """
    y_true, y_pred = [], []
    for item in input_data:
        true_label = str(item.get("True Label")).strip()
        model_answer = str(item.get("model_answer_number")).strip()
        y_true.append(true_label)
        y_pred.append(model_answer)

    acc = accuracy_score(y_true, y_pred)
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    total = len(y_true)
    return {
        "accuracy": round(acc * 100, 2),
        "correct": correct,
        "total": total,
    }


def make_block(df, group_cols, qtypes=("cross", "cultural", "fact", "temporal")):
    """Return (accuracy_table, num_table) for given grouping columns.
    df must contain *_correct and *_num columns for each question type.
    All averages are weighted: sum(correct)/sum(total).
    """
    c_cols = [f"{t}_correct" for t in qtypes]
    n_cols = [f"{t}_num" for t in qtypes]

    g_correct = df.groupby(group_cols, dropna=False)[c_cols].sum()
    g_total = df.groupby(group_cols, dropna=False)[n_cols].sum()

    # Per type weighted accuracy
    acc_mat = safe_div(g_correct.values, g_total.values) * 100
    acc = pd.DataFrame(acc_mat, index=g_correct.index, columns=[f"{t}_accuracy" for t in qtypes]).round(2)

    # Row weighted average across all types
    row_acc = safe_div(g_correct.sum(axis=1).values, g_total.sum(axis=1).values) * 100
    acc["row_avg"] = np.round(row_acc, 2)

    # Column weighted averages
    col_acc_vals = safe_div(g_correct.sum().values, g_total.sum().values) * 100
    col_acc = pd.Series(col_acc_vals, index=[f"{t}_accuracy" for t in qtypes]).round(2)

    overall_acc = safe_div(g_correct.sum().sum(), g_total.sum().sum()) * 100
    col_acc["row_avg"] = round(overall_acc, 2)

    acc.loc["col_avg"] = col_acc

    # Num table
    nums = g_total.copy()
    nums["row_sum"] = nums.sum(axis=1)
    col_sum = nums.sum()
    col_sum.name = "col_sum"
    nums = pd.concat([nums, col_sum.to_frame().T])

    return acc, nums


def make_table(folder_name: Path):
    """Create result.md for a single model-folder.
    Expects JSON files inside `folder_name` named with qtypes (cross/cultural/fact/temporal).
    """
    pattern = re.compile(r"(cultural|cross|fact|temporal)")
    data_dict = {}

    for json_file in folder_name.glob("*.json"):
        match = pattern.search(json_file.name)
        if match is None:
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data_dict[match.group(1)] = data

    if not data_dict:  # no json matched
        return

    # meta info
    character_info = pd.read_json("../../data/source_data/meta_character.json")
    character_info = (
        character_info.reset_index()
        .melt(
            id_vars="index",
            value_vars=["china", "en", "korea", "mexico", "spain", "indonesia"],
            var_name="country",
            value_name="info",
        )
        .dropna(subset=["info"])
        .rename(columns={"index": "character"})
        .reset_index(drop=True)
    )

    character_info_expanded = pd.concat(
        [character_info.drop(columns=["info"]), character_info["info"].apply(pd.Series)], axis=1
    )

    # 1) 각 파일별 accuracy dict 만들기
    result_dict = {}
    for q_type, countries in data_dict.items():
        result_dict[q_type] = {}
        for country, chars in countries.items():
            result_dict[q_type][country] = {}
            for character, rows in chars.items():
                result_dict[q_type][country][character] = compute_accuracy_per_file(rows)

    # 2) dict -> DataFrame (wide)
    result_df = pd.DataFrame.from_dict(
        {
            (q_type, country): characters
            for q_type, countries in result_dict.items()
            for country, characters in countries.items()
        },
        orient="index",
    )
    result_df.index.names = ["Question Type", "Country"]

    # 3) wide -> long
    result_df_reset = (
        result_df.reset_index()
        .melt(id_vars=["Question Type", "Country"], var_name="character_name", value_name="accuracy_info")
        .dropna(subset=["accuracy_info"])
    )

    # 4) accuracy_info dict 분리
    df = result_df_reset.copy()
    df[["accuracy", "correct", "total"]] = df["accuracy_info"].apply(pd.Series)[["accuracy", "correct", "total"]]

    # 5) 캐릭터 × 질문타입 단위 집계 (합/평균)
    agg = (
        df.groupby(["character_name", "Question Type"])
        .agg(
            accuracy=("accuracy", "mean"),
            correct=("correct", "sum"),
            total=("total", "sum"),
        )
        .unstack("Question Type")
    )

    # 6) 컬럼명 정리
    new_cols = []
    for metric, qtype in agg.columns:
        if metric == "accuracy":
            new_cols.append(f"{qtype}_accuracy")
        elif metric == "correct":
            new_cols.append(f"{qtype}_correct")
        else:
            new_cols.append(f"{qtype}_num")
    agg.columns = new_cols
    result = agg.reset_index()

    # 7) 기본 필터링 (cross/temporal NaN 허용 처리용)
    cols_to_check = [c for c in result.columns if ("cross" not in c) and ("temporal" not in c) and c.endswith("_accuracy")]
    filtered_result = result.dropna(subset=cols_to_check)

    # 8) 메타정보 merge
    info_cols = ["character", "country", "history", "time"]
    merged_df = (
        filtered_result.merge(
            character_info_expanded[info_cols],
            left_on="character_name",
            right_on="character",
            how="left",
        ).drop(columns="character")
    )

    # 9) present에서 cross/temporal은 적용 대상이 아니므로 accuracy는 NaN, num/correct는 0으로 맞춤
    ct_cols_acc = [c for c in merged_df.columns if ("cross_" in c or "temporal_" in c) and c.endswith("_accuracy")]
    ct_cols_num = [c for c in merged_df.columns if ("cross_" in c or "temporal_" in c) and c.endswith("_num")]
    ct_cols_cor = [c for c in merged_df.columns if ("cross_" in c or "temporal_" in c) and c.endswith("_correct")]

    mask_present = merged_df["time"] == "present"
    merged_df.loc[mask_present, ct_cols_acc] = np.nan
    merged_df.loc[mask_present, ct_cols_num] = 0
    merged_df.loc[mask_present, ct_cols_cor] = 0

    # 10) NaN 보정: num/correct는 0으로, accuracy는 그대로 NaN 유지해도 됨
    num_cols = [c for c in merged_df.columns if c.endswith("_num")]
    cor_cols = [c for c in merged_df.columns if c.endswith("_correct")]
    merged_df[num_cols] = merged_df[num_cols].fillna(0)
    merged_df[cor_cols] = merged_df[cor_cols].fillna(0)

    # 11) groupings별 테이블 생성
    groupings = {
        "Country": ["country"],
        "History": ["history"],
        "Time": ["time"],
        "History & Time": ["history", "time"],
    }

    markdown_lines = []
    for name, cols in groupings.items():
        acc_tbl, num_tbl = make_block(merged_df, cols)
        markdown_lines.append(f"### Accuracy by {name}\n")
        markdown_lines.append(acc_tbl.to_markdown())
        markdown_lines.append("\n")

        markdown_lines.append(f"### Num by {name}\n")
        markdown_lines.append(num_tbl.to_markdown())
        markdown_lines.append("\n")

    output_path = folder_name / "result.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

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


def count_total_items(character_info, data, question_type):
    """
    총 질문 수(=프로그레스 바 분모)를 스킵 로직을 반영해 계산.
    - cross/fact: country와 character가 data에 존재할 때만 카운트
    - cultural: country 단위로 data[country] 리스트 길이
    - temporal: data 전체 리스트 길이
    """
    total = 0
    if question_type in ["cross", "fact"]:
        for country in character_info:
            if country not in data:
                continue
            for character in character_info[country]:
                if character not in data[country]:
                    continue
                total += len(data[country][character])
    elif question_type == "cultural":
        for country in character_info:
            if country not in data:
                continue
            total += len(data[country])
    elif question_type == "temporal":
        total += len(data)
    return total


def run_mc_evaluation(mc_list, model_name, name, context, template, is_gpt=False,
                      client=None, llm=None, sampling_params=None,
                      batch_size=64, temperature=0.0, seed=42, pbar=None, t0=None):
    """
    pbar: tqdm 객체(전역 1개)
    t0: perf_counter()로 받은 시작 시각(ETA 안정화용 지표 계산)
    """
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

            # 디버깅/추적용: 최근 프롬프트 저장
            with open('./prompt_text.txt', "w", encoding="utf-8") as f:
                f.write("----- PROMPT -----\n")
                f.write(prompt + "\n\n")
                f.write("----- MESSAGES -----\n")
                f.write(json.dumps(messages, ensure_ascii=False, indent=2))

            if "o1" in model_name:
                outputs = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
            else:
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

            if pbar is not None:
                pbar.update(1)
                if t0 is not None:
                    elapsed = perf_counter() - t0
                    qps = pbar.n / max(elapsed, 1e-9)
                    pbar.set_postfix(qps=f"{qps:.2f}", elapsed=f"{elapsed:.0f}s")

    else:
        prompts = []
        for row in mc_list:
            prompt_text = template.format(
                character=name, profile=context,
                Question=row['Question'],
                answer1=row['one'], answer2=row['two'],
                answer3=row['three'], answer4=row['four'], answer5=row['five']
            )
            prompts.append((prompt_text, row))

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            prompt_texts = [p[0] for p in batch]
            rows = [p[1] for p in batch]

            outputs = llm.generate(prompt_texts, sampling_params)

            for row, output in zip(rows, outputs):
                generated_text = output.outputs[0].text.strip()
                model_answer = generated_text
                if "deepseek" in model_name:
                    model_answer = generated_text.split("\n")[-1].strip()
                result_data.append({
                    "Question": row['Question'],
                    "True Label": row['True Label'],
                    "one": row['one'], "two": row['two'], "three": row['three'],
                    "four": row['four'], "five": row['five'],
                    "model_result": generated_text,
                    "model_answer": model_answer,
                    "model_answer_number": extract_model_answer_number(model_answer)
                })

            if pbar is not None:
                pbar.update(len(batch))
                if t0 is not None:
                    elapsed = perf_counter() - t0
                    qps = pbar.n / max(elapsed, 1e-9)
                    pbar.set_postfix(qps=f"{qps:.2f}", elapsed=f"{elapsed:.0f}s")

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

    # ====== NEW: 총 처리 개수 집계 & 글로벌 프로그레스 바 ======
    total_items = count_total_items(character_info, data, args.question_type)
    logger.info(f"Total MC items to evaluate: {total_items}")
    pbar = tqdm(total=total_items, desc="Evaluating", unit="q", smoothing=0.3, dynamic_ncols=True)
    t0_perf = perf_counter()
    # ===========================================================

    result_dic = {}

    if args.device_index:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
        device_indices = args.device_index.split(",")
        tensor_parallel_size = len(device_indices)
    else:
        tensor_parallel_size = 1

    filename = f"{args.question_type}_evaluation_result.json"

    prompt_template_name = args.prompt_template_path.split("mc_eval_template_")[-1]
    prompt_template_name = prompt_template_name.split(".txt")[0]
    output_folder_name = args.model_name.split("/")[-1] if "gpt" not in args.model_name.lower() else args.model_name
    if args.input_dir_path.split("/")[-1] != "test_data":
        output_folder_name = f"{output_folder_name}_{args.input_dir_path.split('/')[-1]}_{prompt_template_name}"

    output_dir = f"../../data/prediction_data/{output_folder_name}/{str(args.context_types)}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Model & template loading (only once)
    is_gpt = any(x in args.model_name.lower() for x in ["gpt", "o1"])
    client = OpenAI(api_key=cfg["openai_key"]) if is_gpt else None
    template_path = args.prompt_template_path
    template = load_template(template_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=512,
        stop=[],
        seed=args.seed
    ) if not is_gpt else None
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=tensor_parallel_size,
        # trust_remote_code=True,  ######################
        dtype="bfloat16"
    ) if not is_gpt else None

    for country in character_info:
        if args.question_type in ["cross", "fact", "cultural"] and country not in data:
            logger.warning(f"No MC data for country '{country}', skipping.")  # 선택
            continue
        result_dic[country] = {}
        for character in character_info[country]:
            if args.question_type in ["cross", "fact"]:
                if country not in data or character not in data[country]:
                    continue
                mc_list_data = data[country][character]
            elif args.question_type == "cultural":
                if country not in data:
                    continue
                mc_list_data = data[country]
            elif args.question_type == "temporal":
                mc_list_data = data

            char_name = character
            char_profile = character_info[country][character]['profile']
            if args.context_types and args.context_types[0] == "no_context":
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
                seed=args.seed,
                pbar=pbar,          # NEW
                t0=t0_perf          # NEW
            )

            result_dic[country][character] = mc_return_list
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dic, f, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=2)

    pbar.close()  # NEW

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    output_folder_dir = f"../../data/prediction_data/{output_folder_name}"
    folder_paths = [folder for folder in Path(output_folder_dir).iterdir() if folder.is_dir()]

    for base_path in folder_paths:
        make_table(base_path)
