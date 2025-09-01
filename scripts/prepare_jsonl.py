#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


SYSTEM_PROMPT = (
    "You are a data analyst that predicts Google Play app installs given app details."
)


def parse_installs(value: str):
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = s.replace(",", "").replace("+", "")
    if s.isdigit():
        return int(s)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def parse_rating(value):
    if pd.isna(value):
        return None
    try:
        f = float(value)
        if f <= 0:
            return None
        return f
    except Exception:
        return None


def build_message(app: str, category: str, rating: float) -> List[dict]:
    user = (
        f"App: {app}\n"
        f"Category: {category}\n"
        f"Rating: {rating if rating is not None else 'unknown'}\n\n"
        f"Predict the approximate installs as an integer."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def row_to_example(row) -> dict:
    app = str(row.get("App", "")).strip()
    category = str(row.get("Category", "")).strip()
    rating = parse_rating(row.get("Rating"))
    installs = parse_installs(row.get("Installs"))
    if not app or not category or installs is None:
        return None
    assistant = str(installs)
    return {"messages": build_message(app, category, rating) + [{"role": "assistant", "content": assistant}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to googleplaystore.csv")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    needed = ["App", "Category", "Rating", "Installs"]
    for col in needed:
        if col not in df.columns:
            print(f"Missing column: {col}", file=sys.stderr)
            sys.exit(1)

    examples = []
    for _, row in df.iterrows():
        ex = row_to_example(row)
        if ex is not None:
            examples.append(ex)

    if len(examples) < 10:
        print("Not enough clean examples to train.", file=sys.stderr)
        sys.exit(1)

    train, val = train_test_split(examples, test_size=args.val_size, random_state=42)

    os.makedirs(os.path.dirname(args.train_jsonl), exist_ok=True)
    with open(args.train_jsonl, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.val_jsonl), exist_ok=True)
    with open(args.val_jsonl, "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train and {len(val)} val examples.")


if __name__ == "__main__":
    main()