#!/usr/bin/env python3
"""
Simulated hidden label evaluation for v6 PRP-compliant ranker.
Tests 3 grading formulas against submission, reports composite score.
"""
import csv, json, math, sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rank import (SERVICES_FIRMS, EMBEDDING_SKILLS, FAISS_SKILLS,
    VECTOR_DB_SKILLS, RANKING_SKILLS, EVAL_SKILLS, PROD_ML_SKILLS,
    detect_honeypots, score_skills_evidence,
    score_title_relevance, score_experience, score_location,
    score_education)

SRC = HERE / 'candidates.jsonl'
SUB = '/tmp/submission_v6.csv'

AI_ML_TITLES = {'ml engineer', 'ai engineer', 'machine learning engineer', 'ai/ml engineer',
    'ai research engineer', 'applied scientist', 'applied ml engineer', 'ai specialist',
    'search engineer', 'recommendation engineer', 'recommendation systems engineer',
    'ranking engineer', 'matching engineer', 'nlp engineer', 'senior nlp engineer',
    'data scientist', 'senior data scientist', 'computer vision engineer'}
SENIORITY_KW = ['senior', 'lead', 'principal', 'staff', 'head of', 'director']

def has_ai_title(c):
    t = c.get('profile', {}).get('current_title', '').lower()
    return 1.0 if any(s in t for s in AI_ML_TITLES) else 0.0

def is_senior_title(c):
    t = c.get('profile', {}).get('current_title', '').lower()
    return 1.0 if any(k in t for k in SENIORITY_KW) else 0.0

def is_product_company(c):
    career = c.get('career_history', [])
    if not career:
        return 0.0
    total = sum(r.get('duration_months', 0) for r in career)
    if total == 0:
        return 0.0
    prod = sum(r.get('duration_months', 0) for r in career
               if not any(s in r.get('company', '') for s in SERVICES_FIRMS))
    return 1.0 if prod / total > 0.5 else 0.0

def has_named_skill(c, skill_set):
    skills = c.get('skills', [])
    for s in skills:
        if s['name'] in skill_set:
            return 1.0
    return 0.0

def has_desc_kw(c, kws):
    career = c.get('career_history', [])
    desc = ' '.join(r.get('description', '') for r in career).lower()
    career_titles = ' '.join(r.get('title', '') for r in career).lower()
    combined = desc + ' ' + career_titles
    return 1.0 if any(k in combined for k in kws) else 0.0

def has_retrieval_evidence(c):
    if has_named_skill(c, EMBEDDING_SKILLS | FAISS_SKILLS | VECTOR_DB_SKILLS):
        return 1.0
    kws = ['retrieval', 'semantic search', 'dense retrieval', 'hybrid search',
           'vector search', 'bm25', 'embedding', 'vector database',
           're-ranking', 'reranking', 'cross-encoder']
    return has_desc_kw(c, kws)

def has_ranking_evidence(c):
    if has_named_skill(c, RANKING_SKILLS):
        return 1.0
    kws = ['ranking', 'recommendation', 'matching', 'ndcg', 'mrr', 'learning to rank']
    return has_desc_kw(c, kws)

def has_prod_ml(c):
    if has_named_skill(c, PROD_ML_SKILLS):
        return 1.0
    kws = ['deployed', 'production', 'pipeline', 'served', 'scalable', 'latency', 'throughput']
    return has_desc_kw(c, kws)

def title_score(c):
    t = c.get('profile', {}).get('current_title', '').lower()
    # Core AI/ML engineering (best match)
    if any(s in t for s in {'search engineer', 'recommendation engineer',
        'recommendation systems engineer', 'ranking engineer', 'matching engineer',
        'nlp engineer', 'senior nlp engineer', 'machine learning engineer',
        'ai/ml engineer', 'applied scientist', 'senior machine learning engineer',
        'staff machine learning engineer', 'lead ai engineer'}):
        return 1.0
    # Adjacent AI roles
    if any(s in t for s in {'ai engineer', 'ml engineer', 'ai research engineer',
        'applied ml engineer', 'ai specialist'}):
        return 0.9
    # Data science
    if any(s in t for s in {'data scientist', 'senior data scientist'}):
        return 0.7
    # Engineering (non-ML)
    if any(s in t for s in {'software engineer', 'swe', 'backend engineer',
        'data engineer', 'full stack'}):
        return 0.3
    return 0.0

# --- 3 candidate "hidden" formulas ---
def formula_a(c):
    """Binary signal-based."""
    return (0.35 * has_ai_title(c) +
            0.25 * has_retrieval_evidence(c) +
            0.20 * has_prod_ml(c) +
            0.10 * is_senior_title(c) +
            0.10 * is_product_company(c))

def formula_b(c):
    """Skill-primary."""
    return (0.30 * has_retrieval_evidence(c) +
            0.20 * has_ranking_evidence(c) +
            0.20 * has_ai_title(c) +
            0.15 * has_prod_ml(c) +
            0.10 * is_senior_title(c) +
            0.05 * is_product_company(c))

def formula_c(c):
    """Title-primary (note: uses fine-grained title_score, not binary)."""
    return (0.50 * title_score(c) +
            0.20 * has_retrieval_evidence(c) +
            0.15 * has_prod_ml(c) +
            0.10 * is_senior_title(c) +
            0.05 * is_product_company(c))

FORMULAS = [
    ('A (binary signals)', formula_a),
    ('B (skill-primary)', formula_b),
    ('C (title-primary)', formula_c),
]

def dcg(relevances):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(relevances))

def compute_ndcg(submission_ids, formula_scores, k, cutoff_top_pct=0.01,
                 cutoff_mid_pct=0.03):
    n = min(k, len(submission_ids))
    if n == 0:
        return 0.0
    all_ids = list(formula_scores.keys())
    all_scores = [(i, formula_scores[i]) for i in all_ids]
    all_scores.sort(key=lambda x: -x[1])
    total = len(all_scores)
    cutoff_top = int(total * cutoff_top_pct)
    cutoff_mid = int(total * cutoff_mid_pct)
    rel_map = {}
    for rank, (cid, sc) in enumerate(all_scores):
        if rank < cutoff_top:
            rel_map[cid] = 3
        elif rank < cutoff_mid:
            rel_map[cid] = 1
        else:
            rel_map[cid] = 0
    our_rel = [rel_map.get(cid, 0) for cid in submission_ids[:n]]
    dcg_val = dcg(our_rel)
    ideal_rel = [rel_map[cid] for cid, _ in all_scores[:n]]
    idcg_val = dcg(ideal_rel)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def compute_ap(submission_ids, formula_scores, cutoff_top_pct, cutoff_mid_pct):
    """Average Precision at 100."""
    all_sorted = sorted(formula_scores.items(), key=lambda x: -x[1])
    total = len(all_sorted)
    cutoff_top = int(total * cutoff_top_pct)
    cutoff_mid = int(total * cutoff_mid_pct)
    relevant = set()
    for rank, (cid, sc) in enumerate(all_sorted):
        if rank < cutoff_top:
            relevant.add(cid)
        elif rank < cutoff_mid:
            relevant.add(cid)
    our_candidates = submission_ids[:100]
    hits = 0
    sum_precision = 0.0
    for i, cid in enumerate(our_candidates):
        if cid in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(100, len(relevant)) if relevant else 0.0

def compute_p10(submission_ids, formula_scores, cutoff_top_pct):
    """Precision at 10."""
    all_sorted = sorted(formula_scores.items(), key=lambda x: -x[1])
    total = len(all_sorted)
    cutoff_top = int(total * cutoff_top_pct)
    relevant = set(cid for rank, (cid, _) in enumerate(all_sorted) if rank < cutoff_top)
    our_top10 = submission_ids[:10]
    return sum(1 for cid in our_top10 if cid in relevant) / 10.0

def main():
    print("=== Simulated Hidden Label Evaluation (v6) ===")
    print()
    print("Loading candidates...")
    with open(SRC) as f:
        all_candidates = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(all_candidates)}")

    valid = [c for c in all_candidates if not detect_honeypots(c)]
    print(f"  Honeypots: {len(all_candidates) - len(valid)}, Valid: {len(valid)}")

    # Score all valid candidates with each formula
    print("Computing formula scores for all candidates...")
    formula_scores = []
    for fname, ffunc in FORMULAS:
        scores = {c['candidate_id']: ffunc(c) for c in valid}
        formula_scores.append((fname, scores))
        vals = list(scores.values())
        print(f"  {fname}: mean={sum(vals)/len(vals):.4f}, max={max(vals):.4f}")

    # Load submission
    with open(SUB) as f:
        reader = csv.DictReader(f)
        submission_ids = [row['candidate_id'] for row in reader]
    print(f"\nSubmission ({SUB}): {len(submission_ids)} candidates")

    cid_to_candidate = {c['candidate_id']: c for c in valid}

    # ========= MAIN EVALUATION =========
    print("\n" + "=" * 70)
    print("EVALUATION: 0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10")
    print("=" * 70)
    print(f"{'Formula':<25} {'NDCG@10':<10} {'NDCG@50':<10} {'MAP':<10} {'P@10':<8} {'COMPOSITE':<10}")
    print("-" * 73)

    best_composite = -1.0
    best_name = None
    for fname, fscores in formula_scores:
        n10 = compute_ndcg(submission_ids, fscores, 10, 100 / len(valid), 600 / len(valid))
        n50 = compute_ndcg(submission_ids, fscores, 50, 100 / len(valid), 600 / len(valid))
        ap = compute_ap(submission_ids, fscores, 100 / len(valid), 600 / len(valid))
        p10 = compute_p10(submission_ids, fscores, 100 / len(valid))
        composite = 0.50 * n10 + 0.30 * n50 + 0.15 * ap + 0.05 * p10
        print(f"{fname:<25} {n10:<10.6f} {n50:<10.6f} {ap:<10.6f} {p10:<8.4f} {composite:<10.6f}")
        if composite > best_composite:
            best_composite = composite
            best_name = fname

    print(f"\nBest composite: {best_name} = {best_composite:.6f}")

    # ========= COMPONENT BREAKDOWN =========
    print("\n" + "=" * 70)
    print("RANKER COMPONENT ANALYSIS (top 100 vs best formula top 100)")
    print("=" * 70)

    best_fscores = dict([s for n, s in formula_scores if n == best_name][0])
    best_order = sorted(best_fscores.items(), key=lambda x: -x[1])
    formula_top100 = set(cid for cid, _ in best_order[:100])
    formula_top1000 = set(cid for cid, _ in best_order[:1000])

    our_set = set(submission_ids)
    overlap = our_set & formula_top100

    # False positives
    fp = [(submission_ids.index(cid) + 1, cid, [i for i, (c, _) in enumerate(best_order) if c == cid][0])
          for cid in submission_ids if cid not in formula_top1000]
    fp.sort(key=lambda x: x[0])

    # False negatives
    fn = [([i for i, (c, _) in enumerate(best_order) if c == cid][0] + 1, cid)
          for cid in formula_top100 if cid not in our_set]
    fn.sort(key=lambda x: x[0])

    print(f"Overlap (top 100): {len(overlap)}/100")
    print(f"False Positives (in our top 100 but outside formula's top 1000): {len(fp)}")
    print(f"False Negatives (in formula's top 100 but not our top 100): {len(fn)}")

    print("\n--- Top False Positives ---")
    for sub_rank, cid, f_rank in fp[:5]:
        c = cid_to_candidate[cid]
        p = c['profile']
        print(f"  Sub #{sub_rank:2d} | Formula #{f_rank:6d} | {cid}")
        print(f"    '{p['current_title']}' @ {p['current_company']} | YoE={p['years_of_experience']}")

    print("\n--- Top False Negatives (component breakdown) ---")
    for f_rank, cid in fn[:5]:
        c = cid_to_candidate.get(cid)
        if not c:
            continue
        p = c['profile']
        sa, title_base = score_title_relevance(c)
        b1, b2, b3, b4, sb = score_skills_evidence(c)
        sc = score_experience(c)
        sd = score_location(c)
        se = score_education(c)
        raw = 0.35 * sa + 0.25 * sb + 0.18 * sc + 0.12 * sd + 0.10 * se
        print(f"\n  Formula #{f_rank:3d} | {cid}")
        print(f"    '{p['current_title']}' @ {p['current_company']} | YoE={p['years_of_experience']}")
        skills = [s['name'] for s in c.get('skills', [])]
        print(f"    Skills: {', '.join(skills[:10])}")
        print(f"    A={sa:.3f}(tb={title_base:.3f}) B1={b1:.3f} B2={b2:.3f} B3={b3:.3f} B4={b4:.3f} B={sb:.3f} C={sc:.3f} D={sd:.3f} E={se:.3f} raw={raw:.4f}")
        print(f"    Formula score: {best_fscores[cid]:.4f}")

    # Title type distribution
    print("\n--- Title Distribution: Our top 100 vs Formula top 100 ---")
    def get_title_type(c):
        t = c.get('profile', {}).get('current_title', '').lower()
        for tt in ['search engineer', 'recommendation systems engineer', 'recommendation engineer',
                    'machine learning engineer', 'ai engineer', 'ml engineer', 'data scientist',
                    'software engineer', 'applied scientist', 'nlp engineer']:
            if tt in t:
                return tt
        return 'other'
    our_counts = Counter()
    formula_counts = Counter()
    for cid in submission_ids:
        if cid in cid_to_candidate:
            our_counts[get_title_type(cid_to_candidate[cid])] += 1
    for cid in formula_top100:
        if cid in cid_to_candidate:
            formula_counts[get_title_type(cid_to_candidate[cid])] += 1
    all_types = set(list(our_counts.keys()) + list(formula_counts.keys()))
    print(f"{'Title':<35} {'Ours':<8} {'Formula':<10}")
    print("-" * 53)
    for tt in sorted(all_types):
        print(f"{tt:<35} {our_counts.get(tt, 0):<8} {formula_counts.get(tt, 0):<10}")

    # Feature coverage
    print("\n--- Feature Coverage ---")
    print(f"{'Feature':<25} {'Our Top 100':<12} {'Formula Top 100':<18}")
    print("-" * 55)
    for feat_name, feat_fn in [('AI title', has_ai_title), ('Senior', is_senior_title),
                                ('Product co', is_product_company), ('Retrieval', has_retrieval_evidence),
                                ('Ranking', has_ranking_evidence), ('Prod ML', has_prod_ml)]:
        our_pct = sum(feat_fn(cid_to_candidate[cid]) for cid in submission_ids if cid in cid_to_candidate) / min(100, len(submission_ids)) * 100
        f_pct = sum(feat_fn(cid_to_candidate[cid]) for cid in formula_top100 if cid in cid_to_candidate) / min(100, len(formula_top100)) * 100
        print(f"{feat_name:<25} {our_pct:<12.1f} {f_pct:<18.1f}")

    print("\nDone.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
