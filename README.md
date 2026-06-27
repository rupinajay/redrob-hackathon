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

- Runtime: ~14s for 100K candidates on CPU
- Memory: ~200MB
- No GPU required, no network calls

## Files

- `rank.py` — Ranking logic
- `validate_submission.py` — Official format validator
- `candidate_schema.json` — Data schema reference
- `submission_metadata.yaml` — Team and submission metadata
