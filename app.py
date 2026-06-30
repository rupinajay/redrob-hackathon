#!/usr/bin/env python3
"""Redrob Ranker sandbox — upload candidates.jsonl, get top 100 rankings."""
import json
import csv
import io
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rank

CONFIG_PATH = Path(__file__).parent / "config.json"


def rank_candidates(file_bytes):
    cfg = rank.load_config(str(CONFIG_PATH))

    # Parse uploaded file
    try:
        text = file_bytes.decode("utf-8")
        candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception as e:
        return f"Error parsing candidates.jsonl: {e}", None, None, None

    if len(candidates) == 0:
        return "No candidates found in file.", None, None, None

    # Compute TF-IDF
    tfidf = rank.compute_tfidf_scores(candidates, cfg)

    # Rank
    out, all_scored = rank.rank_candidates(candidates, cfg, tfidf)

    # Build output CSV
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in out:
        w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    csv_text = buf.getvalue()

    # Build summary
    scores = [r["score"] for r in out]
    summary = (
        f"Processed {len(candidates)} candidates. "
        f"Top score: {scores[0]:.4f}, Bottom score: {scores[-1]:.4f}. "
        f"Unique scores: {len(set(scores))}. "
        f"Unique reasonings: {len(set(r['reasoning'] for r in out))}."
    )

    # Build HTML table for top 20
    html_rows = ""
    for r in out[:20]:
        reason = r["reasoning"][:120]
        html_rows += (
            f"<tr><td>{r['rank']}</td><td>{r['candidate_id']}</td>"
            f"<td>{r['score']:.6f}</td><td>{reason}</td></tr>"
        )
    html_table = f"""<div style="max-height:500px;overflow-y:auto;font-size:13px">
    <table border="1" cellpadding="6" cellspacing="0" style="width:100%;border-collapse:collapse">
    <thead><tr style="background:#eee">
    <th>Rank</th><th>Candidate</th><th>Score</th><th>Reasoning</th>
    </tr></thead><tbody>{html_rows}</tbody></table></div>"""

    return summary, csv_text, html_table, candidates


with gr.Blocks(title="Redrob Candidate Ranker", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # 🏆 Redrob Hackathon — Candidate Ranker
    Upload a `candidates.jsonl` file to get the top 100 rankings for the **Senior AI Engineer** role.

    ### How to use
    1. Upload your `candidates.jsonl` file
    2. The ranker processes all candidates and returns the top 100
    3. View the top 20 in the table below
    4. Download the full submission.csv
    """
    )

    with gr.Row():
        file_input = gr.File(label="Upload candidates.jsonl", file_types=[".jsonl"])
        run_btn = gr.Button("🚀 Rank Candidates", variant="primary", size="lg")

    summary = gr.Textbox(label="Summary", lines=2, interactive=False)
    results_table = gr.HTML(label="Top 20 Candidates")

    download = gr.File(label="Download submission.csv", interactive=False)

    all_candidates_state = gr.State()

    def process(file):
        if file is None:
            return "Please upload a file.", "", None, None
        with open(file.name, "rb") as f:
            data = f.read()
        summary, csv_text, html_table, candidates = rank_candidates(data)

        # Write CSV to temp file for download
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        tmp.write(csv_text)
        tmp.close()

        return summary, html_table, tmp.name, candidates

    run_btn.click(
        fn=process,
        inputs=[file_input],
        outputs=[summary, results_table, download, all_candidates_state],
    )

    gr.Markdown(
        """
    ---
    **Compute**: CPU-only, no GPU/network required. Runtime ~30s per 100K candidates.
    **Repo**: [github.com/rupinajay/redrob-hackathon](https://github.com/rupinajay/redrob-hackathon)
    """
    )


if __name__ == "__main__":
    demo.launch()
