#!/usr/bin/env python3
"""Redrob Ranker — upload candidates.jsonl, get top 100 rankings."""
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

CUSTOM_CSS = """
:root {
  --primary: #1a1a2e;
  --primary-light: #16213e;
  --accent: #0f3460;
  --accent-light: #e94560;
  --bg: #f8f9fc;
  --card-bg: #ffffff;
  --text: #1a1a2e;
  --text-muted: #6b7280;
  --border: #e5e7eb;
  --success: #059669;
  --warning: #d97706;
  --radius: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
}

body {
  background: var(--bg);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
  padding: 1.5rem 1rem !important;
  background: transparent !important;
}

.app-header {
  text-align: center;
  margin-bottom: 2rem;
}

.app-title {
  font-size: 2.2rem;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: -0.02em;
  margin: 0 0 0.3rem 0;
}

.app-subtitle {
  font-size: 1.05rem;
  color: var(--text-muted);
  font-weight: 400;
  margin: 0;
}

.card {
  background: var(--card-bg);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  padding: 1.5rem;
  margin-bottom: 1.25rem;
  transition: box-shadow 0.2s ease;
}

.card:hover {
  box-shadow: var(--shadow-lg);
}

.card-header {
  font-size: 1rem;
  font-weight: 600;
  color: var(--primary);
  margin: 0 0 1rem 0;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}

.card-header small {
  font-weight: 400;
  color: var(--text-muted);
  font-size: 0.85rem;
}

.upload-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.upload-area {
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.5rem !important;
  text-align: center !important;
  transition: all 0.2s ease !important;
  background: #fafbfc !important;
}

.upload-area:hover {
  border-color: var(--accent) !important;
  background: #f0f4ff !important;
}

.upload-area label {
  font-weight: 500 !important;
  color: var(--text) !important;
  font-size: 0.95rem !important;
}

.run-btn {
  background: var(--primary) !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.65rem 1.5rem !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  color: white !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.01em !important;
}

.run-btn:hover {
  background: var(--accent) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(15, 52, 96, 0.3) !important;
}

.summary-box {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: #f0fdf4 !important;
  font-size: 0.9rem !important;
}

.summary-box label {
  font-weight: 600 !important;
  color: var(--success) !important;
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

.results-table thead th {
  background: var(--primary);
  color: white;
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  position: sticky;
  top: 0;
  z-index: 10;
}

.results-table thead th:first-child {
  border-radius: 6px 0 0 0;
}

.results-table thead th:last-child {
  border-radius: 0 6px 0 0;
}

.results-table tbody tr {
  border-bottom: 1px solid #f0f0f0;
  transition: background 0.15s ease;
}

.results-table tbody tr:hover {
  background: #f5f7ff;
}

.results-table tbody tr:nth-child(even) {
  background: #fafbfc;
}

.results-table tbody tr:nth-child(even):hover {
  background: #f0f4ff;
}

.results-table tbody td {
  padding: 10px 14px;
  vertical-align: top;
  color: var(--text);
}

.rank-cell {
  font-weight: 700;
  color: var(--accent);
  width: 50px;
}

.id-cell {
  font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace;
  font-size: 0.82rem;
  color: var(--accent);
  width: 130px;
}

.score-cell {
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--primary);
  width: 90px;
}

.reasoning-cell {
  font-size: 0.84rem;
  line-height: 1.45;
  color: var(--text);
}

.download-btn {
  background: var(--success) !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  color: white !important;
  transition: all 0.2s ease !important;
}

.download-btn:hover {
  filter: brightness(1.1) !important;
  transform: translateY(-1px) !important;
}

.app-footer {
  text-align: center;
  padding: 1.5rem 0 0.5rem;
  border-top: 1px solid var(--border);
  margin-top: 1.5rem;
}

.footer-text {
  font-size: 0.82rem;
  color: var(--text-muted);
}

.footer-text a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}

.footer-text a:hover {
  text-decoration: underline;
}

@media (max-width: 640px) {
  .app-title { font-size: 1.6rem; }
  .card { padding: 1rem; }
}
"""


def rank_candidates(file_bytes):
    cfg = rank.load_config(str(CONFIG_PATH))

    try:
        text = file_bytes.decode("utf-8")
        candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception as e:
        return f"Error parsing candidates.jsonl: {e}", None, None, None

    if len(candidates) == 0:
        return "No candidates found in file.", None, None, None

    tfidf = rank.compute_tfidf_scores(candidates, cfg)

    out, all_scored = rank.rank_candidates(candidates, cfg, tfidf)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in out:
        w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    csv_text = buf.getvalue()

    scores = [r["score"] for r in out]
    top_titles = {}
    for r in out:
        t = r["reasoning"].split(" with ")[0].strip() if " with " in r["reasoning"] else ""
        top_titles[t] = top_titles.get(t, 0) + 1
    top_role = max(top_titles, key=top_titles.get) if top_titles else "N/A"

    summary = (
        f"Processed {len(candidates)} candidates. "
        f"Score range: {scores[-1]:.4f} - {scores[0]:.4f}. "
        f"Unique scores: {len(set(scores))}. "
        f"Top role represented: {top_role} ({top_titles[top_role]} of 100)."
    )

    html_rows = ""
    for r in out[:20]:
        reason = r["reasoning"][:150]
        rank_class = "rank-cell"
        if r["rank"] <= 3:
            rank_class += " medal"
        html_rows += (
            f"<tr>"
            f"<td class='rank-cell'>{r['rank']}</td>"
            f"<td class='id-cell'>{r['candidate_id']}</td>"
            f"<td class='score-cell'>{r['score']:.6f}</td>"
            f"<td class='reasoning-cell'>{reason}</td>"
            f"</tr>"
        )

    html_table = f"""<div class='card' style='padding:0;overflow:hidden'>
    <div style='max-height:520px;overflow-y:auto'>
    <table class='results-table'>
    <thead><tr>
    <th>Rank</th><th>Candidate ID</th><th>Score</th><th>Reasoning</th>
    </tr></thead>
    <tbody>{html_rows}</tbody>
    </table></div></div>"""

    return summary, csv_text, html_table, candidates


with gr.Blocks(
    title="Redrob Candidate Ranker",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        """
    <div class='app-header'>
      <h1 class='app-title'>Redrob Candidate Ranker</h1>
      <p class='app-subtitle'>Upload a candidates.jsonl file to rank the top 100 candidates for the Senior AI Engineer role.</p>
    </div>
    """
    )

    with gr.Column():
        with gr.Column(elem_classes="card"):
            gr.HTML("<div class='card-header'>Upload Candidates</div>")
            with gr.Row():
                file_input = gr.File(label="", file_types=[".jsonl"], elem_classes="upload-area")
                run_btn = gr.Button("Rank Candidates", variant="primary", size="lg", elem_classes="run-btn")

        summary = gr.Textbox(
            label="",
            lines=2,
            interactive=False,
            elem_classes="summary-box",
            show_label=False,
        )

        results_table = gr.HTML("", visible=True)

        with gr.Column(elem_classes="card"):
            gr.HTML("<div class='card-header'>Download Full Results</div>")
            download = gr.File(
                label="",
                interactive=False,
                show_label=False,
                elem_classes="download-btn",
            )

    all_candidates_state = gr.State()

    def process(file):
        if file is None:
            return "Please upload a candidates.jsonl file to begin.", None, None, None
        with open(file.name, "rb") as f:
            data = f.read()
        summary, csv_text, html_table, candidates = rank_candidates(data)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        tmp.write(csv_text)
        tmp.close()
        return summary, html_table, tmp.name, candidates

    run_btn.click(
        fn=process,
        inputs=[file_input],
        outputs=[summary, results_table, download, all_candidates_state],
    )

    gr.HTML(
        """
    <div class='app-footer'>
      <p class='footer-text'>
        CPU-only &middot; No GPU required &middot; Runtime ~30s per 100K candidates &middot;
        <a href='https://github.com/rupinajay/redrob-hackathon' target='_blank'>GitHub</a>
      </p>
    </div>
    """
    )


if __name__ == "__main__":
    demo.launch()
