
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extend cross-choices with third-person factual options, with tqdm progress.

Adds:
- Global tqdm progress bar over all QA transformations (attempted).
- Pretty elapsed-time summary at the end.
"""

import argparse
import json
import os
import re
import sys
import time
import yaml
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy

# tqdm (optional but recommended)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# OpenAI SDK
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # allow import even if not installed; runtime will error if used


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def infer_pronoun_set_from_text(text: str) -> str:
    """
    Heuristic: return 'she' | 'he' | 'they'
    Looks for gendered keywords in the provided text (profile/context/summary).
    """
    t = (text or "").lower()
    female_markers = [
        "actress", "queen", "princess", "her ", " she ", "female", "woman", "girl",
        "empress", "daughter", "singer-actress", "lady"
    ]
    male_markers = [
        "actor", "king", "prince", " his ", " he ", "male", "man", "boy",
        "emperor", "son"
    ]
    # quick checks
    if any(m in t for m in female_markers):
        return "she"
    if any(m in t for m in male_markers):
        return "he"
    return "they"


def build_pronoun_lookup(pronoun_key: str) -> Dict[str, str]:
    """
    Map key -> actual pronoun forms for the target referent.
    """
    if pronoun_key == "she":
        return dict(subject="she", object="her", poss_det="her", poss_pron="hers", reflexive="herself")
    if pronoun_key == "he":
        return dict(subject="he", object="him", poss_det="his", poss_pron="his", reflexive="himself")
    # default neutral
    return dict(subject="they", object="them", poss_det="their", poss_pron="theirs", reflexive="themselves")


def extract_json_blob(text: str) -> str:
    """
    Extract the first top-level JSON object from a string. Helps if the
    model wraps output in code fences or adds minor text.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text  # best effort


def call_gpt_transform(
    client: "OpenAI",
    qa: Dict[str, str],
    referent_name: str,
    pronouns: Dict[str, str],
    speaker_name: str,
    model: str = "gpt-4o-mini",
    retries: int = 3,
    sleep: float = 1.2,
) -> Dict[str, str]:
    """
    Ask the model to convert the given QA (first-person by referent_name)
    into third-person (about referent_name) as answered by speaker_name.
    Only person/pronouns change; everything else must remain identical.
    """
    sys_msg = (
        "You are a precise text transformer. "
        "You will receive a JSON object with keys 'Question', 'Answer', "
        "and 'Incorrect Answer 1'..'Incorrect Answer 10' (some may be missing). "
        "The text is written in first person by PERSON_X. "
        "Transform it into third person about PERSON_X, as if PERSON_Y is answering about PERSON_X. "
        "STRICT REQUIREMENTS:\n"
        "1) In 'Question', replace any 'you/your/yours/yourself' that refer to PERSON_X so that the question asks about PERSON_X **by proper name** exactly as provided (do not use pronouns in the question).\n"
        "2) In all answers, replace only first-person references to PERSON_X ('I, me, my, mine, myself') with the provided third-person pronouns for PERSON_X.\n"
        "3) Do not change **any other words**, numbers, dates, punctuation, capitalization, or spacing; do not fix typos.\n"
        "4) Preserve the same keys as input and the same count of answer options.\n"
        "5) Preserve 'I can not answer that question.' exactly if it appears.\n"
        "6) Output only a JSON object. No extra commentary."
    )

    user_msg = {
        "instruction": {
            "person_x_name": referent_name,
            "person_x_pronouns": pronouns,  # subject/object/poss_det/poss_pron/reflexive
            "person_y_name": speaker_name
        },
        "input": qa
    }

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)}
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            blob = extract_json_blob(text)
            out = json.loads(blob)
            # sanity checks
            for k in qa.keys():
                if k not in out:
                    out[k] = qa[k]
            return out
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(sleep * (attempt + 1))
    return qa


def get_country_chars(meta: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns per-country lists of 'past' and 'present' character names.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for country, entries in meta.items():
        past, present = [], []
        for name, payload in entries.items():
            t = (payload or {}).get("time", "").strip().lower()
            (past if t == "past" else present if t == "present" else present).append(name)
        out[country] = {"past": past, "present": present}
    return out


def existing_question_set(cross_list: List[Dict[str, str]]) -> set:
    qset = set()
    for item in cross_list:
        q = (item or {}).get("Question", "")
        if q:
            qset.add(q.strip())
    return qset


def choose_pronoun_for_character(char_name: str, meta_entry: Dict[str, Any], overrides: Dict[str, str]) -> Dict[str, str]:
    # 1) explicit override
    if overrides and char_name in overrides:
        return build_pronoun_lookup(overrides[char_name].strip().lower())
    # 2) infer from text fields
    text = " ".join([
        str(meta_entry.get("profile", "") or ""),
        str(meta_entry.get("context", {}).get("Summary", "") or ""),
        str(meta_entry.get("context", {}).get("name", "") or "")
    ])
    key = infer_pronoun_set_from_text(text)
    return build_pronoun_lookup(key)


def compute_total_tasks(meta: Dict[str, Any], cross: Dict[str, Any], facts: Dict[str, Any],
                        country_filter: Optional[set], limit_facts: int) -> int:
    """
    Pre-compute how many QA items will be *attempted* (for tqdm total).
    This counts fact QAs per (country, past, present), honoring limit_facts.
    """
    total = 0
    country_chars = get_country_chars(meta)
    for country, groups in country_chars.items():
        if country_filter and country not in country_filter:
            continue
        past_names = groups.get("past", []) or []
        present_names = groups.get("present", []) or []
        if country not in cross or country not in facts:
            continue
        for past in past_names:
            for pres in present_names:
                pres_facts = facts.get(country, {}).get(pres, [])
                if not pres_facts:
                    continue
                if limit_facts and limit_facts > 0:
                    total += min(limit_facts, len(pres_facts))
                else:
                    total += len(pres_facts)
    return total


def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML with openai_key")
    ap.add_argument("--meta", required=True, help="meta_character.json")
    ap.add_argument("--cross", required=True, help="cise_cross_choices_version3_check_fin.json")
    ap.add_argument("--facts", required=True, help="cise_fact_choices_version3_check_fin.json")
    ap.add_argument("--out", required=True, help="Output path for extended cross JSON")
    ap.add_argument("--countries", default="", help="Comma-separated allowlist of countries (optional)")
    ap.add_argument("--limit_facts", type=int, default=0, help="Max facts per present character (0 = all)")
    ap.add_argument("--dry_run", action="store_true", help="Do not write output file")
    ap.add_argument("--pronoun_map", default="", help="JSON with {'Character Name': 'she|he|they', ...}")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for transformation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if OpenAI is None:
        print("ERROR: openai SDK not installed. `pip install openai`", file=sys.stderr)
        sys.exit(2)
    client = OpenAI(api_key=cfg["openai_key"])

    meta = load_json(args.meta)
    cross = load_json(args.cross)
    facts = load_json(args.facts)

    country_filter = set([c.strip() for c in args.countries.split(",") if c.strip()]) if args.countries else None

    # load pronoun overrides if provided
    overrides: Dict[str, str] = {}
    if args.pronoun_map and os.path.exists(args.pronoun_map):
        try:
            overrides = load_json(args.pronoun_map)
        except Exception:
            print("WARN: failed to load pronoun_map; continuing with heuristics.", file=sys.stderr)

    country_chars = get_country_chars(meta)

    # Pre-compute total tasks for tqdm
    total_tasks = compute_total_tasks(meta, cross, facts, country_filter, args.limit_facts)
    use_tqdm = (tqdm is not None) and (total_tasks > 0)

    start_time = time.time()
    added_total = 0
    per_country_stats: Dict[str, int] = {}

    pbar = tqdm(total=total_tasks, desc="Extending cross with facts", unit="QA") if use_tqdm else None

    for country, groups in country_chars.items():
        if country_filter and country not in country_filter:
            continue

        past_names = groups.get("past", []) or []
        present_names = groups.get("present", []) or []

        # skip countries not present in both cross and facts (we only add where both exist)
        if country not in cross or country not in facts:
            continue

        # build fast lookup for pronouns of present characters
        pronoun_cache: Dict[str, Dict[str, str]] = {}
        for pres in present_names:
            meta_entry = meta.get(country, {}).get(pres, {}) or {}
            pronoun_cache[pres] = choose_pronoun_for_character(pres, meta_entry, overrides)

        per_country_added = 0

        for past in past_names:
            # ensure list exists
            cross.setdefault(country, {})
            cross[country].setdefault(past, [])
            bucket = cross[country][past]
            existing_qs = existing_question_set(bucket)

            speaker_name = past

            for pres in present_names:
                pres_facts = facts.get(country, {}).get(pres, [])
                if not pres_facts:
                    continue

                pronouns = pronoun_cache[pres]
                limit = args.limit_facts if args.limit_facts and args.limit_facts > 0 else len(pres_facts)
                used = 0

                for qa in pres_facts:
                    if used >= limit:
                        break

                    # Transform via GPT: person/pronoun only.
                    transformed = call_gpt_transform(
                        client=client,
                        qa=qa,
                        referent_name=pres,
                        pronouns=pronouns,
                        speaker_name=speaker_name,
                        model=args.model
                    )

                    qtext = transformed.get("Question", "").strip()
                    if qtext:
                        if qtext not in existing_qs:
                            bucket.append(transformed)
                            existing_qs.add(qtext)
                            added_total += 1
                            per_country_added += 1

                    used += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(f"{country} | {past} → {pres} | added={added_total}")

        if per_country_added:
            per_country_stats[country] = per_country_stats.get(country, 0) + per_country_added

    if pbar is not None:
        pbar.close()

    elapsed = time.time() - start_time

    if args.dry_run:
        print(f"[DRY-RUN] New items that would be added: {added_total}")
        print(f"[DRY-RUN] Countries processed: {sorted(list((country_filter or set(country_chars.keys()))))}")
        print(f"Elapsed: {format_elapsed(elapsed)}")
        return

    # Write output (never overwrite input)
    if os.path.abspath(args.out) == os.path.abspath(args.cross):
        # In-place not allowed: safeguard
        root, ext = os.path.splitext(args.out)
        args.out = root + ".extended" + ext

    save_json(args.out, cross)
    print(f"Done. Added {added_total} new items.")
    print(f"Wrote: {args.out}")
    if per_country_stats:
        print("Per-country added:")
        for c, n in sorted(per_country_stats.items()):
            print(f"  - {c}: {n}")
    print(f"Elapsed: {format_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
