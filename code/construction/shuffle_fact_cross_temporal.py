import os
import json
import argparse
import pandas as pd
import random
from tqdm import tqdm

def read_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data

def shuffle_mc_answers(data, type):

    result_data = []

    for row in data:


        answer = row["Answer"]

        fixed_choices = [answer]
        choice_num = 1
        
        check_real_fact = False

        # cross factual 처리
        if "cross" in type:
            if "Incorrect Answer 9" in row:
                real_fact = row["Incorrect Answer 9"]
                check_real_fact = True
                fixed_choices.append(real_fact)
                choice_num+=1
        
        # factual 처리
        elif "fact" in type:
            fixed_choices.append(row["Incorrect Answer 9"])
            choice_num+=1
    
        group1_num = 0
        group2_num = 0

        # 오답 보기 9개 있는 경우
        if "Incorrect Answer 9" in row:
            group1_num, group2_num = 1, 1

        # 오답 보기 8개 있는 경우
        else:
            group1_num, group2_num = 2, 2

        choice_num+=group1_num
        choice_num+=group2_num

        # Incorrect Answer 1~4 중에서 group1_num개, 5~8 중에서 group2_num개 선택
        group1 = []
        group2 = []
        for i in range(1, 9):
            key = f"Incorrect Answer {i}"
            if key in row and row[key] not in fixed_choices:
                if i <= 4:
                    group1.append(row[key])
                else:
                    group2.append(row[key])
        sampled = []
        group1 = []
        group2 = []
        for i in range(1, 9):
            key = f"Incorrect Answer {i}"
            if key in row and row[key] not in fixed_choices:
                if i <= 4:
                    group1.append(row[key])
                else:
                    group2.append(row[key])

        sampled = []

        # group1에서 group1_num개 추출
        if len(group1) >= group1_num:
            sampled.extend(random.sample(group1, group1_num))
        elif group1:  # 개수가 부족하면 있는 것만 다 사용
            sampled.extend(group1)
            print(f"[Warn] Not enough distractors in group1 for: {row['Question']} "
                  f"(needed {group1_num}, got {len(group1)})")
        else:
            print(f"[Warn] No valid distractor in group1 for: {row['Question']}")

        # group2에서 group2_num개 추출
        if len(group2) >= group2_num:
            sampled.extend(random.sample(group2, group2_num))
        elif group2:
            sampled.extend(group2)
            print(f"[Warn] Not enough distractors in group2 for: {row['Question']} "
                  f"(needed {group2_num}, got {len(group2)})")
        else:
            print(f"[Warn] No valid distractor in group2 for: {row['Question']}")


        if choice_num <5:
            # 총 5개가 되도록 추가로 하나 더 뽑기 (남은 distractors에서)
            used_set = set(fixed_choices + sampled)
            remaining = [row[f"Incorrect Answer {i}"] for i in range(1, 9)
                        if f"Incorrect Answer {i}" in row and row[f"Incorrect Answer {i}"] not in used_set]

            if remaining:
                sampled.append(random.choice(remaining))
            else:
                print(f"[Warn] No remaining distractors to fill for: {row['Question']}")

        all_choices = fixed_choices + sampled
        random.shuffle(all_choices)

        correct_index = all_choices.index(answer) + 1

        if check_real_fact:
            entry = {
                "Question": row["Question"],
                "Answer": answer,
                "True Label": correct_index,
                "one": all_choices[0],
                "two": all_choices[1],
                "three": all_choices[2],
                "four": all_choices[3],
                "five": all_choices[4],
                "real_fact": real_fact
            }

        else:
            entry = {
                "Question": row["Question"],
                "Answer": answer,
                "True Label": correct_index,
                "one": all_choices[0],
                "two": all_choices[1],
                "three": all_choices[2],
                "four": all_choices[3],
                "five": all_choices[4],
            }
        result_data.append(entry)

    return result_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_type", type=str, help="Prefix for file naming (choice: temporal, fact, cross)")
    parser.add_argument("--prompt_version", type=str, default="original")
    parser.add_argument("--sample_choice", type=bool, default=False)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    data = read_json(f'../../data/source_data/{args.question_type}_choices_{args.prompt_version}.json')

    if args.question_type == 'temporal':
        output_data = shuffle_mc_answers(data, args.question_type)
    else:
        output_data = {}
        for country in data:
            output_data[country] = {}
            for character in data[country]:
                temp_output = shuffle_mc_answers(data[country][character], args.question_type)
                output_data[country][character] = temp_output

    if args.output_folder:
        output_file = f'../../data/{args.output_folder}/{args.question_type}_mc.json'
    else:
        output_file = f"../../data/test_data_{args.prompt_version}/{args.question_type}_mc.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Saved shuffled results to {output_file}")