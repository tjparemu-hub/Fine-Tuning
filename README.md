### Google Play Installs Fine-Tuning (OpenAI)

Prereqs: Python 3.9+.

1) Install deps

```bash
pip3 install -r requirements.txt
```

2) Prepare data (adjust the CSV path as needed)

```bash
python3 scripts/prepare_jsonl.py \
  --csv "$HOME/Downloads/googleplaystore.csv" \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl
```

3) Set your API key

```bash
export OPENAI_API_KEY=YOUR_KEY
```

4) Start fine-tune and monitor

```bash
python3 scripts/fine_tune.py \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --suffix installs-predictor
```

After success, the job output will contain the fine-tuned model name. Use it like:

```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
  model="ft:gpt-4o-mini:org:installs-predictor:xyz",
  messages=[
    {"role": "system", "content": "You are a data analyst that predicts Google Play app installs given app details."},
    {"role": "user", "content": "App: MyApp\nCategory: TOOLS\nRating: 4.5\n\nPredict the approximate installs as an integer."}
  ]
)
print(completion.choices[0].message)
```