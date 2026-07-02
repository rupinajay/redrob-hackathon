#!/usr/bin/env python3
"""Redrob Ranker — upload candidates, preview data, rank the top 100."""
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
  max-width: 960px; margin: 0 auto; padding: 1.5rem 1rem 2rem;
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

progress, .progress-level {
  height: 8px !important;
  border-radius: 4px !important;
}

.pt {
  width: 100%; border-collapse: collapse;
  font-size: 0.82rem;
}
.pt thead th {
  background: var(--surface-hover);
  color: var(--text-muted);
  font-size: 0.68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 7px 10px; text-align: left;
  border-bottom: 1px solid var(--border);
}
.pt tbody td {
  padding: 7px 10px; vertical-align: top;
  border-bottom: 1px solid var(--surface-hover);
  color: var(--text-secondary);
}
.pt tbody tr:last-child td { border-bottom: none; }
.pt .pi { font-family: 'SF Mono', monospace; color: var(--text-muted); font-size: 0.76rem; }
.pt .ptt { color: var(--text); font-weight: 500; }
.pt .pl { font-size: 0.78rem; }

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

.rank-btn { margin-top: 0.25rem; }
.rank-btn button {
  width: 100% !important;
  background: var(--primary) !important;
  color: #000 !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.6rem 1rem !important;
  font-size: 0.88rem !important;
  cursor: pointer !important;
}
.rank-btn button:hover { opacity: 0.9 !important; }
"""


def parse_candidates(raw_bytes):
    text = raw_bytes.decode("utf-8").strip()
    # Try JSONL first (one object per line)
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        try:
            objs = [json.loads(l) for l in lines]
            # Verify they look like candidate objects
            if all(isinstance(o, dict) and "candidate_id" in o for o in objs):
                return objs
        except json.JSONDecodeError:
            pass
    # Try JSON array
    try:
        objs = json.loads(text)
        if isinstance(objs, list) and all(isinstance(o, dict) and "candidate_id" in o for o in objs):
            return objs
    except json.JSONDecodeError:
        pass
    return None


def build_preview(candidates):
    n = len(candidates)
    rows = ""
    for c in candidates[:5]:
        p = c.get("profile", {})
        cid = c.get("candidate_id", "?")
        title = p.get("current_title", "?")
        yoe = p.get("years_of_experience", "?")
        loc = f"{p.get('location', '?')}, {p.get('country', '?')}"
        headline = (p.get("headline", "") or "")[:70]
        rows += (
            f"<tr>"
            f"<td class='pi'>{cid}</td>"
            f"<td class='ptt'>{title}</td>"
            f"<td>{yoe}</td>"
            f"<td class='pl'>{loc}</td>"
            f"<td class='pl'>{headline}</td>"
            f"</tr>"
        )

    more = f"<div style='color:var(--text-muted);font-size:0.8rem;padding:7px 10px'>... and {n - 5} more candidates</div>" if n > 5 else ""

    return (
        f"<div style='padding:0.5rem 0 0.25rem 0;color:var(--text);font-size:0.88rem;font-weight:600'>"
        f"{n:,} candidates loaded</div>"
        f"<div class='tw'><table class='pt'><thead><tr>"
        f"<th style='width:120px'>ID</th><th style='width:140px'>Title</th>"
        f"<th style='width:52px'>YOE</th><th style='width:180px'>Location</th><th>Headline</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>{more}</div>"
    )


with gr.Blocks(
    title="Redrob Candidate Ranker",
    fill_height=True,
) as demo:
    gr.HTML(
        '<div class="hdr"><h1>Redrob Candidate Ranker</h1>'
        "<p>Upload candidate data to rank the top 100 for the Senior AI Engineer role.</p></div>"
    )

    gr.HTML('<div class="mn">')

    # Upload card
    with gr.Column(elem_classes="cd"):
        gr.HTML('<div class="cd-t">Upload candidates</div>')
        file_input = gr.UploadButton(
            "Upload candidates.json / .jsonl",
            file_types=[".jsonl", ".json"],
            file_count="single",
            variant="secondary",
            elem_classes="up-btn",
        )

    # Preview card (hidden until file uploaded)
    preview_html = gr.HTML(visible=False)

    # Rank button (hidden until preview shown)
    rank_btn = gr.Button("Rank Candidates", variant="primary", visible=False, elem_classes="rank-btn")

    # How it works
    gr.HTML(
        '<div class="cd">'
        '<div class="cd-t">How it works</div>'
        '<ul style="margin:0;padding-left:1.2rem;font-size:0.86rem;color:var(--text-secondary);line-height:1.7">'
        "<li>Upload a <code>candidates.jsonl</code> or <code>candidates.json</code> file from the Redrob challenge dataset</li>"
        "<li>Review a preview of the loaded candidates before ranking</li>"
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
        '    api_name="/rank"\n'
        ")\n"
        "<span style='color:#64748b'># result[0] = stats HTML, result[1] = CSV path</span>"
        "</pre></div>"
    )

    # Results area
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

    gr.HTML(
        '<div class="ft">'
        "CPU-only  ·  No GPU  ·  ~30s per 100K candidates  ·  "
        '<a href="https://github.com/rupinajay/redrob-hackathon" target="_blank">GitHub</a>'
        "</div>"
    )

    # State to hold parsed candidates across steps
    candidates_state = gr.State()

    def load_and_preview(file):
        if file is None:
            return "", gr.update(visible=False), gr.update(visible=False), None

        path = file if isinstance(file, str) else file.path
        with open(path, "rb") as f:
            raw = f.read()

        candidates = parse_candidates(raw)
        if candidates is None:
            err = "<div style='color:#ef4444;padding:0.75rem 0'>Error: could not parse file. Expected .jsonl (one JSON per line) or .json (array).</div>"
            return err, gr.update(visible=False), gr.update(visible=False), None
        if len(candidates) == 0:
            err = "<div style='color:#f59e0b;padding:0.75rem 0'>No candidates found in file.</div>"
            return err, gr.update(visible=False), gr.update(visible=False), None

        preview = build_preview(candidates)
        return (
            preview,
            gr.update(visible=True),
            gr.update(visible=True),
            candidates,
        )

    def rank_candidates(candidates, progress=gr.Progress(track_tqdm=True)):
        if not candidates:
            return "", gr.update(visible=False), None

        progress(0.05, desc="Loading config...")
        cfg = rank.load_config(str(CONFIG_PATH))

        progress(0.1, desc=f"Computing TF-IDF ({len(candidates):,} candidates)...")
        tfidf_scores = rank.compute_tfidf_scores(candidates, cfg, progress=progress)

        progress(0.5, desc="Ranking candidates...")
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
        fn=load_and_preview,
        inputs=[file_input],
        outputs=[preview_html, preview_html, rank_btn, candidates_state],
    )

    rank_btn.click(
        fn=rank_candidates,
        inputs=[candidates_state],
        outputs=[results_html, dl_card, download_btn],
    )


if __name__ == "__main__":
    demo.queue().launch(css=CSS)
