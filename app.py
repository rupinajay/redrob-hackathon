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

# ── Brand palette ──
# Primary: Indigo (trust, intelligence, tech)
# Accent: Amber (warmth, energy, action)
# Surface: White on cool-gray background
# Text: Dark slate with semantic hierarchy

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f4f5f7;
  --surface: #ffffff;
  --surface-hover: #f8f9fc;
  --border: #e2e4e9;
  --border-light: #eef0f4;
  --primary: #4f46e5;
  --primary-dark: #3730a3;
  --primary-light: #eef2ff;
  --primary-bg: rgba(79, 70, 229, 0.04);
  --accent: #f59e0b;
  --accent-bg: #fffbeb;
  --text: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --success: #059669;
  --success-bg: #ecfdf5;
  --success-border: #a7f3d0;
  --radius: 10px;
  --radius-sm: 6px;
  --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 1px 1px rgba(0,0,0,0.02);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.04), 0 2px 4px -1px rgba(0,0,0,0.02);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.04), 0 4px 6px -2px rgba(0,0,0,0.02);
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'SF Mono', 'SFMono-Regular', ui-monospace, monospace;
}

body {
  margin: 0; padding: 0;
  background: var(--bg);
  font-family: var(--font);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

.gradio-container {
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
}

/* ── Header ── */
.hdr {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  padding: 3rem 1.5rem 2.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.hdr::after {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 30%, rgba(255,255,255,0.08) 0%, transparent 60%);
  pointer-events: none;
}
.hdr h1 {
  font-size: 2rem; font-weight: 700; color: #fff;
  letter-spacing: -0.03em; margin: 0 0 0.5rem 0; position: relative;
}
.hdr p {
  font-size: 1rem; color: rgba(255,255,255,0.7); font-weight: 400;
  margin: 0; position: relative;
}

/* ── Main ── */
.mn {
  max-width: 940px; margin: 0 auto; padding: 1.5rem 1rem 2rem;
}

/* ── Card ── */
.cd {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow);
}
.cd-t {
  font-size: 0.75rem; font-weight: 600; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin: 0 0 0.75rem 0;
}

/* ── Upload row ── */
.upload-r {
  display: flex; gap: 0.75rem; align-items: stretch;
}
.upload-r .gr-file {
  flex: 1 !important;
}
.upload-r .gr-file label {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 1.5rem 1rem !important;
  margin: 0 !important;
  background: var(--surface) !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  min-height: 48px !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
}
.upload-r .gr-file label:hover {
  border-color: var(--primary) !important;
  background: var(--primary-light) !important;
  color: var(--primary) !important;
}
.upload-r .gr-file input { display: none !important; }

.run-btn-wrap button {
  height: 100% !important;
  min-height: 48px !important;
  padding: 0 1.5rem !important;
  background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: #fff !important;
  font-family: var(--font) !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  white-space: nowrap !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}
.run-btn-wrap button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(79,70,229,0.35) !important;
}

/* ── Summary ── */
.sm {
  padding: 0.75rem 1rem;
  background: var(--success-bg);
  border: 1px solid var(--success-border);
  border-radius: var(--radius-sm);
  color: #065f46;
  font-size: 0.88rem;
  font-weight: 500;
}

/* ── Results table ── */
.tw {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.rt {
  width: 100%; border-collapse: collapse;
  font-size: 0.86rem;
}
.rt thead th {
  background: #f8f9fc;
  color: var(--text-secondary);
  font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 9px 12px; text-align: left;
  border-bottom: 1px solid var(--border);
}
.rt tbody tr {
  border-bottom: 1px solid var(--border-light);
  transition: background 0.1s;
}
.rt tbody tr:last-child { border-bottom: none; }
.rt tbody tr:hover { background: var(--primary-bg); }
.rt tbody td { padding: 9px 12px; vertical-align: top; }
.rt td.r { font-weight: 700; color: var(--primary); }
.rt td.i { font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted); }
.rt td.s { font-weight: 600; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text); }
.rt td.rs { font-size: 0.82rem; line-height: 1.5; color: var(--text-secondary); }

/* ── Download ── */
.dl-r {
  display: flex; align-items: center; gap: 0.75rem;
}
.dl-r label {
  display: inline-flex !important;
  align-items: center !important;
  gap: 0.4rem !important;
  padding: 0.5rem 1.1rem !important;
  background: var(--primary) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: #fff !important;
  font-family: var(--font) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  margin: 0 !important;
}
.dl-r label:hover {
  background: var(--primary-dark) !important;
  transform: translateY(-1px);
}

/* ── Footer ── */
.ft {
  text-align: center; padding: 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem; color: var(--text-muted);
}
.ft a { color: var(--primary); text-decoration: none; font-weight: 500; }
.ft a:hover { text-decoration: underline; }
"""


def rank_candidates(file_bytes):
    cfg = rank.load_config(str(CONFIG_PATH))
    try:
        text = file_bytes.decode("utf-8")
        candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception as e:
        return f"Error: {e}", "", ""
    if not candidates:
        return "No candidates found.", "", ""

    tfidf = rank.compute_tfidf_scores(candidates, cfg)
    out, _ = rank.rank_candidates(candidates, cfg, tfidf)

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

    stats = (
        f"{len(candidates)} candidates processed  ·  "
        f"Score range {scores[-1]:.4f} - {scores[0]:.4f}  ·  "
        f"{len(set(scores))} unique scores  ·  "
        f"Top role: {top_role} ({top_titles.get(top_role, 0)}x)"
    )

    rows = ""
    for r in out[:20]:
        rows += (
            f"<tr><td class='r'>{r['rank']}</td>"
            f"<td class='i'>{r['candidate_id']}</td>"
            f"<td class='s'>{r['score']:.6f}</td>"
            f"<td class='rs'>{r['reasoning'][:160]}</td></tr>"
        )
    table = (
        '<div class="tw"><table class="rt"><thead><tr>'
        '<th style="width:52px">Rank</th><th style="width:130px">Candidate</th>'
        '<th style="width:92px">Score</th><th>Reasoning</th>'
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    )

    return stats, table, csv_text


with gr.Blocks(
    title="Redrob Candidate Ranker",
    fill_height=True,
) as demo:
    # Header
    gr.HTML(
        '<div class="hdr"><h1>Redrob Candidate Ranker</h1>'
        "<p>Upload candidates.jsonl to rank the top 100 for the Senior AI Engineer role.</p></div>"
    )

    gr.HTML('<div class="mn">')

    # Upload card
    with gr.Column(elem_classes="cd"):
        gr.HTML('<div class="cd-t">Upload candidates</div>')
        with gr.Row(elem_classes="upload-r"):
            file_input = gr.File(
                label="",
                show_label=False,
                file_types=[".jsonl"],
                scale=4,
            )
            run_btn = gr.Button(
                "Rank Candidates",
                variant="primary",
                elem_classes="run-btn-wrap",
                scale=1,
            )

    # Summary (hidden until results come)
    summary_html = gr.HTML('<div class="sm">Awaiting upload...</div>')

    # Results table (hidden initially)
    results_html = gr.HTML("")

    # Download card (hidden until results come)
    with gr.Column(elem_classes="cd", visible=False) as dl_card:
        gr.HTML('<div class="cd-t">Download results</div>')
        with gr.Row(elem_classes="dl-r"):
            gr.HTML(
                '<span style="font-size:0.88rem;color:var(--text-secondary);flex:1">'
                "Full submission.csv — all 100 ranked candidates.</span>"
            )
            download_btn = gr.File(
                label="Download submission.csv",
                show_label=False,
                file_types=[".csv"],
                interactive=False,
            )

    gr.HTML("</div>")

    # Footer
    gr.HTML(
        '<div class="ft">'
        "CPU-only  ·  No GPU  ·  ~30s per 100K candidates  ·  "
        '<a href="https://github.com/rupinajay/redrob-hackathon" target="_blank">GitHub</a>'
        "</div>"
    )

    def process(file):
        if file is None:
            sm = '<div class="sm">Upload a candidates.jsonl file to begin.</div>'
            return sm, "", gr.update(visible=False), None
        path = file if isinstance(file, str) else file.path
        with open(path, "rb") as f:
            data = f.read()
        stats, table, csv_text = rank_candidates(data)
        sm = f'<div class="sm">{stats}</div>'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        tmp.write(csv_text)
        tmp.close()
        return sm, table, gr.update(visible=True), tmp.name

    run_btn.click(
        fn=process,
        inputs=[file_input],
        outputs=[summary_html, results_html, dl_card, download_btn],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
