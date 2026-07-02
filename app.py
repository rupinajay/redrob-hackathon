#!/usr/bin/env python3
"""Redrob Ranker — upload candidates.jsonl, get top 100 rankings."""
import json
import csv
import io
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rank

CONFIG_PATH = Path(__file__).parent / "config.json"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #000000;
  --surface: #0a0a0a;
  --surface-hover: #111111;
  --border: #1f1f1f;
  --primary: #818cf8;
  --text: #e2e8f0;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
}

body {
  margin: 0; padding: 0;
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif;
  color: var(--text);
}

.gradio-container {
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: var(--bg) !important;
}

.hdr {
  background: linear-gradient(135deg, #1e1b4b 0%, #0f0f0f 100%);
  padding: 3rem 1.5rem 2.5rem;
  text-align: center;
  border-bottom: 1px solid var(--border);
}
.hdr h1 {
  font-size: 2rem; font-weight: 700; color: var(--primary);
  letter-spacing: -0.03em; margin: 0 0 0.5rem 0;
}
.hdr p {
  font-size: 1rem; color: var(--text-muted); margin: 0;
}

.mn {
  max-width: 940px; margin: 0 auto; padding: 1.5rem 1rem 2rem;
}

.cd {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}
.cd-t {
  font-size: 0.75rem; font-weight: 600; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin: 0 0 0.75rem 0;
}

.up-btn { width: 100% !important; }
.up-btn button {
  width: 100% !important;
  border: 2px dashed var(--border) !important;
  border-radius: 6px !important;
  padding: 1.75rem 1rem !important;
  background: var(--surface) !important;
  color: var(--text-muted) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
}
.up-btn button:hover {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
}

.ft {
  text-align: center; padding: 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem; color: var(--text-muted);
}
.ft a { color: var(--primary); text-decoration: none; font-weight: 500; }

/* ── Progress bar ── */
progress, .progress-level {
  height: 8px !important;
  border-radius: 4px !important;
}

/* ── Results table ── */
.tw {
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
.rt {
  width: 100%; border-collapse: collapse;
  font-size: 0.86rem;
}
.rt thead th {
  background: var(--surface-hover);
  color: var(--text-muted);
  font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 9px 12px; text-align: left;
  border-bottom: 1px solid var(--border);
}
.rt tbody tr {
  border-bottom: 1px solid var(--surface-hover);
}
.rt tbody tr:last-child { border-bottom: none; }
.rt tbody td { padding: 9px 12px; vertical-align: top; }
.rt td.r { font-weight: 700; color: var(--primary); }
.rt td.i { font-family: 'SF Mono', monospace; font-size: 0.78rem; color: var(--text-muted); }
.rt td.s { font-weight: 600; font-family: 'SF Mono', monospace; font-size: 0.8rem; color: var(--text); }
.rt td.rs { font-size: 0.82rem; line-height: 1.5; color: var(--text-secondary); }
"""


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
        file_input = gr.UploadButton(
            "Upload candidates.jsonl",
            file_types=[".jsonl"],
            file_count="single",
            variant="secondary",
            elem_classes="up-btn",
        )

    # How it works
    gr.HTML(
        '<div class="cd">'
        '<div class="cd-t">How it works</div>'
        '<ul style="margin:0;padding-left:1.2rem;font-size:0.86rem;color:var(--text-secondary);line-height:1.7">'
        "<li>Upload a <code>candidates.jsonl</code> file from the Redrob challenge dataset</li>"
        "<li>Each candidate is scored against the Senior AI Engineer job description</li>"
        "<li>Scoring uses TF-IDF semantic matching, skill evidence, career trajectory,"
        " education, and location signals</li>"
        "<li>The top 100 candidates are ranked with unique scores and reasoning text</li>"
        "<li>Download the full <code>submission.csv</code> for submission</li>"
        "</ul></div>"
    )

    # API usage
    gr.HTML(
        '<div class="cd">'
        '<div class="cd-t">Use via API</div>'
        '<pre style="background:#111;border:1px solid #1f1f1f;border-radius:6px;'
        'padding:1rem;font-size:0.78rem;line-height:1.6;overflow-x:auto;color:#94a3b8">'
        '<span style="color:#64748b"># pip install gradio-client</span>\n'
        "from gradio_client import Client, handle_file\n\n"
        'client = Client("https://rupinajay-redrob-ranker.hf.space")\n'
        'result = client.predict(\n'
        '    handle_file("candidates.jsonl"),\n'
        '    api_name="/process"\n'
        ")\n"
        "<span style='color:#64748b'># result[0] = stats HTML, result[1] = results CSV path</span>"
        "</pre></div>"
    )

    # Results
    results_html = gr.HTML(visible=False)

    # Download
    with gr.Column(elem_classes="cd", visible=False) as dl_card:
        gr.HTML('<div class="cd-t">Download results</div>')
        with gr.Row():
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

    def process(file, progress=gr.Progress(track_tqdm=True)):
        if file is None:
            return "", gr.update(visible=False), None

        path = file if isinstance(file, str) else file.path

        progress(0.02, desc="Loading file...")
        with open(path, "rb") as f:
            data = f.read()

        progress(0.05, desc="Loading config...")
        cfg = rank.load_config(str(CONFIG_PATH))

        progress(0.08, desc="Parsing candidates...")
        try:
            text = data.decode("utf-8")
            candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
        except Exception as e:
            return f"<div style='color:#ef4444'>Error: {e}</div>", gr.update(visible=False), None
        if not candidates:
            return "<div style='color:#f59e0b'>No candidates found.</div>", gr.update(visible=False), None

        progress(0.1, desc=f"Computing TF-IDF ({len(candidates):,} candidates)...")
        tfidf_scores = rank.compute_tfidf_scores(candidates, cfg, progress=progress)

        progress(0.5, desc=f"Ranking candidates...")
        out, _ = rank.rank_candidates(candidates, cfg, tfidf_scores, progress=progress)

        progress(0.9, desc="Building output...")

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

        summary = (
            f"<div style='padding:0.75rem 1rem;background:#052e16;"
            f"border:1px solid #166534;border-radius:6px;"
            f"color:#34d399;font-size:0.88rem;font-weight:500;"
            f"margin-bottom:0.75rem'>{stats}</div>"
        )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        tmp.write(csv_text)
        tmp.close()

        progress(1.0, desc="Complete")
        return summary + table, gr.update(visible=True), tmp.name

    file_input.upload(
        fn=process,
        inputs=[file_input],
        outputs=[results_html, dl_card, download_btn],
    )


if __name__ == "__main__":
    demo.queue().launch(css=CSS)
