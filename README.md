---
title: Redrob Candidate Ranker
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
---

# Redrob Hackathon — Candidate Ranking

Ranks the top 100 candidates from `candidates.jsonl` for the Senior AI Engineer role at Redrob AI.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Validate output

```bash
python validate_submission.py ./submission.csv
```

## Compute

- Runtime: ~30s for 100K candidates on CPU
- Memory: <1GB
- No GPU required, no network calls

## Files

- `rank.py` — Config-driven ranking logic
- `config.json` — All weights, thresholds, and rules
- `validate_submission.py` — Official format validator
- `candidate_schema.json` — Data schema reference
- `submission_metadata.yaml` — Team and submission metadata
