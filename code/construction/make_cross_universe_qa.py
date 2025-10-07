import json
import random
import pandas as pd
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=False, default="./data/source_data/meta_character.json",
                    help="Directory where the character JSON file is located")
args = parser.parse_args()

def make_cross_qa_df(json_file):

    templates = [
            "Do you know {name}?",
            "Have you heard of {name}?",
            "Have you met {name}?"
        ]
    
    df = pd.DataFrame(columns=['country','character', 'question', 'answer'])

    rows = []
    for country in json_file:
        for character in json_file[country]:
            for other in json_file[country]:
                if other == character:
                    continue
                if json_file[country][other]["time"] == "present" and json_file[country][character]["time"]=="past":
                    q = random.choice(templates).format(name=other)
                    a = "I can not answer that question."

                    rows.append({
                        "country" : country,
                        "character" : character,
                        "profile" : json_file[country][character]["profile"],
                        "question": q,
                        "answer": a
                    })
    
    if rows:
        df = pd.DataFrame(rows)

    return df



if __name__ == "__main__":

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        characters = json.load(f)

    result_df = make_cross_qa_df(json_file=characters)

    if "sample" in args.input_dir:
        output_dir = f'{args.input_dir.split('meta_character')[0]}cross_universe_qa_sample.csv'
    else:
        output_dir = f'{args.input_dir.split('meta_character')[0]}cross_universe_qa.csv'

    result_df.to_csv(output_dir, encoding='utf-8', index=False)
    