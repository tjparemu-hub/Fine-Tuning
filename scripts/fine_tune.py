#!/usr/bin/env python3
import argparse
import os
import time
from typing import Optional

from openai import OpenAI
from rich import print


def upload_file(client: OpenAI, path: str, purpose: str = "fine-tune") -> str:
    with open(path, "rb") as f:
        file = client.files.create(file=f, purpose=purpose)
    return file.id


def create_job(client: OpenAI, train_file: str, val_file: Optional[str], suffix: str) -> str:
    job = client.fine_tuning.jobs.create(
        training_file=train_file,
        validation_file=val_file,
        model="gpt-4o-mini",
        suffix=suffix,
    )
    print(job)
    return job.id


def wait_for_completion(client: OpenAI, job_id: str):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"status: {status}")
        if status in {"succeeded", "failed", "cancelled"}:
            print(job)
            break
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=False)
    parser.add_argument("--suffix", default="installs-predictor")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    train_id = upload_file(client, args.train_jsonl)
    val_id = upload_file(client, args.val_jsonl) if args.val_jsonl else None

    job_id = create_job(client, train_id, val_id, args.suffix)

    wait_for_completion(client, job_id)


if __name__ == "__main__":
    main()