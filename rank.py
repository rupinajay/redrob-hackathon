#!/usr/bin/env python3
"""
Redrob Hackathon Ranker v5.0 — Data-Driven Ranker

Architecture:
  1. Title gate (pre-filter): non-engineering titles capped at 0.30
  2. Continuous title scoring (no hard archetype tiers)
  3. Retrieval evidence + product company + behavioral as main signals
  4. Behavioral restored to moderate range (data shows AI/ML candidates
     have 2x open_to_work and 40% higher response rates than rest)
"""
import argparse, csv, json, sys, math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config (inline; role-agnostic by design)
# ---------------------------------------------------------------------------
SERVICES_FIRMS = {'Wipro', 'TCS', 'Infosys', 'Accenture', 'Cognizant', 'Capgemini',
    'HCL', 'Tech Mahindra', 'Mphasis', 'Mindtree', 'L&T Infotech', 'Hexaware',
    'Persistent', 'NIIT Technologies', 'Mastech'}

SENIORITY_KW = ['senior', 'lead', 'principal', 'staff', 'head of', 'director']
JUNIOR_KW = ['junior', 'associate', 'trainee', 'intern', 'fresher']

# Title base scores (continuous, no hard tiers)
AI_CORE_TITLES = {'ml engineer', 'ai engineer', 'machine learning engineer', 'ai/ml engineer',
    'ai research engineer', 'applied scientist', 'applied ml engineer', 'ai specialist'}
SEARCH_TITLES = {'search engineer', 'recommendation engineer', 'recommendation systems engineer',
    'ranking engineer', 'matching engineer'}
NLP_TITLES = {'nlp engineer', 'senior nlp engineer'}
DATA_SCI_TITLES = {'data scientist', 'senior data scientist'}
DATA_ENG_TITLES = {'data engineer', 'senior data engineer', 'analytics engineer'}
COMP_SCI_TITLES = {'computer vision engineer'}
GEN_ENG_TITLES = {'software engineer', 'swe', 'backend engineer', 'frontend engineer',
    'frontend developer', 'full stack', 'full stack developer', 'devops engineer',
    'devops', 'qa engineer', 'qa', 'mobile developer', 'cloud engineer',
    'systems engineer', 'java developer', '.net developer'}
NON_ENG_KW = ['hr', 'operations', 'marketing', 'accountant', 'mechanical', 'civil',
    'graphic designer', 'project manager', 'business analyst', 'customer support',
    'sales executive', 'content writer', 'sales']

# Retrieval evidence
# Retrieval evidence — split into subcategories per JD
RETRIEVAL_SKILLS = {'BM25', 'FAISS', 'Annoy', 'sentence-transformers', 'BGE', 'E5',
    'dense retrieval', 'sparse retrieval', 'hybrid search', 'vector search', 'embedding'}
VECTOR_DB_SKILLS = {'Pinecone', 'Qdrant', 'Weaviate', 'Milvus', 'Chroma', 'Elasticsearch', 'OpenSearch'}
RANKING_SKILLS = {'NDCG', 'MRR', 'MAP', 'learning to rank', 'LambdaRank',
    'recommendation systems', 'recommender', 'matching', 're-ranking'}
EVAL_SKILLS = {'A/B testing', 'offline evaluation', 'ranking evaluation', 'evaluation', 'metrics'}
PROD_ML_SKILLS = {'PyTorch', 'TensorFlow', 'scikit-learn', 'Hugging Face', 'transformers',
    'Keras', 'Langchain', 'FastAPI', 'Flask', 'Docker', 'Kubernetes', 'CI/CD',
    'MLOps', 'Airflow', 'Kubeflow', 'GCP', 'AWS', 'Azure', 'Spark', 'Python'}

RETRIEVAL_DESC_KW = ['retrieval', 'semantic search', 'dense retrieval', 'hybrid search',
    'vector search', 'bm25', 'embedding', 'vector database', 're-ranking', 'reranking',
    'cross-encoder']
RANKING_DESC_KW = ['ranking', 'recommendation', 'matching', 'ndcg', 'mrr', 'learning to rank']
EVAL_DESC_KW = ['evaluation', 'offline eval', 'ab testing', 'a/b testing', 'metrics']
PROD_DESC_KW = ['deployed', 'production', 'pipeline', 'served', 'scalable', 'latency', 'throughput']

PREFERRED_CITIES = {'noida', 'pune', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'gurgaon'}

JD_TFIDF_TEXT = " ".join([
    "Senior AI Engineer retrieval ranking recommendation systems information retrieval",
    "embedding models sentence-transformers BGE E5 FAISS vector embeddings",
    "vector databases Pinecone Qdrant Weaviate Milvus Chroma Elasticsearch",
    "evaluation metrics NDCG MRR MAP precision recall ranking quality",
    "production ML Python PyTorch TensorFlow scikit-learn FastAPI",
    "semantic search neural search hybrid search cross-encoder bi-encoder",
    "candidate matching resume parsing talent discovery search infrastructure",
    "sparse retrieval dense retrieval BM25 re-ranking reranking",
    "recommender systems product engineering mindset scale",
])

# ---------------------------------------------------------------------------
# Title scoring
# ---------------------------------------------------------------------------
def get_title_base(title_lower):
    for s in SEARCH_TITLES:
        if s in title_lower: return 0.95
    for s in AI_CORE_TITLES:
        if s in title_lower: return 0.95
    for s in NLP_TITLES:
        if s in title_lower: return 0.90
    for s in DATA_SCI_TITLES:
        if s in title_lower: return 0.70
    for s in COMP_SCI_TITLES:
        if s in title_lower: return 0.70
    for s in DATA_ENG_TITLES:
        if s in title_lower: return 0.50
    for s in GEN_ENG_TITLES:
        if s in title_lower: return 0.30
    if any(k in title_lower for k in NON_ENG_KW):
        return 0.10
    return 0.10

def get_seniority_mult(title_lower):
    if any(k in title_lower for k in SENIORITY_KW): return 1.0
    if any(k in title_lower for k in JUNIOR_KW): return 0.75
    return 0.90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_product_prop(career):
    if not career: return 0.0
    total = sum(r.get('duration_months', 0) for r in career)
    if total == 0: return 0.0
    prod = sum(r.get('duration_months', 0) for r in career
               if not any(s in r.get('company', '') for s in SERVICES_FIRMS))
    return prod / total

def skill_trust(skill):
    p = skill.get('proficiency', 'beginner')
    d = skill.get('duration_months', 0)
    e = skill.get('endorsements', 0)
    base = {'beginner': 0.25, 'intermediate': 0.55, 'advanced': 0.80, 'expert': 1.00}.get(p, 0.25)
    if d == 0 and p in ('advanced', 'expert'): return 0.0
    return base * min(1.0, d / 24) * min(1.15, 1.0 + e / 120)

# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------
def score_title_relevance(candidate):
    t = candidate.get('profile', {}).get('current_title', '').lower()
    base = get_title_base(t)
    mult = get_seniority_mult(t)
    return min(1.0, base * mult), base

def score_skills_evidence(candidate):
    """Return (retrieval, ranking, evaluation, production, composite) subscores."""
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    desc = ' '.join(r.get('description', '') for r in career).lower()
    career_titles = ' '.join(r.get('title', '') for r in career).lower()

    def best(skill_set, trust_mul=0.85):
        b = 0.0
        for s in skills:
            if s['name'] in skill_set:
                b = max(b, min(1.0, trust_mul * skill_trust(s)))
        return b

    def desc_density(kws, max_score=0.35):
        mentions = 0
        for r in career:
            d = (r.get('description', '') or '').lower()
            mentions += sum(d.count(k) for k in kws)
        if mentions == 0:
            return 0.0
        return min(max_score, max_score * math.sqrt(mentions) / 3.0)

    def title_density(kws, max_score=0.25):
        mentions = sum(career_titles.count(k) for k in kws)
        if mentions == 0:
            return 0.0
        return min(max_score, max_score * math.sqrt(mentions) / 3.0)

    r = max(best(RETRIEVAL_SKILLS | VECTOR_DB_SKILLS, 0.85), desc_density(RETRIEVAL_DESC_KW, 0.35), title_density(RETRIEVAL_DESC_KW, 0.25))
    rk = max(best(RANKING_SKILLS, 0.85), desc_density(RANKING_DESC_KW, 0.35), title_density(RANKING_DESC_KW, 0.25))
    e = max(best(EVAL_SKILLS, 0.80), desc_density(EVAL_DESC_KW, 0.30))
    p = max(best(PROD_ML_SKILLS, 0.70), desc_density(PROD_DESC_KW, 0.25))
    composite = 0.35 * r + 0.30 * rk + 0.20 * p + 0.15 * e
    return r, rk, e, p, composite

def score_product_experience(candidate):
    career = candidate.get('career_history', [])
    prop = get_product_prop(career)
    all_services = all(any(s in r.get('company', '') for s in SERVICES_FIRMS) for r in career) if career else False
    penalty = -0.20 if all_services else 0.0
    bonus = 0.10 if prop > 0.5 else 0.0
    title = candidate.get('profile', {}).get('current_title', '').lower()
    seniority = 0.05 if any(k in title for k in SENIORITY_KW) else 0.0
    return min(1.0, max(0.0, prop + bonus + penalty + seniority))

def score_yoe(candidate):
    yoe = candidate.get('profile', {}).get('years_of_experience', 0)
    if 5 <= yoe <= 9: return 1.0
    if 4 <= yoe < 5 or 9 < yoe <= 11: return 0.82
    if 3 <= yoe < 4 or 11 < yoe <= 14: return 0.58
    return 0.25

def score_location(candidate):
    p = candidate.get('profile', {})
    r = candidate.get('redrob_signals', {})
    country = p.get('country', '')
    location = p.get('location', '').lower()
    relo = r.get('willing_to_relocate', False)
    pref = any(c in location for c in PREFERRED_CITIES)
    if country == 'India':
        return 1.0 if pref else (0.85 if relo else 0.55)
    return 0.30 if relo else 0.05

def score_education(candidate):
    edu = candidate.get('education', [])
    if not edu: return 0.20
    best = max(edu, key=lambda e: e.get('end_year', 0))
    tier = best.get('tier', 'unknown')
    field = best.get('field_of_study', '').lower()
    tech = any(f in field for f in {'computer science', 'cse', 'cs', 'it', 'ai', 'ml', 'electronics', 'math', 'statistics'})
    scores = {'tier_1': (1.0, 0.60), 'tier_2': (0.82, 0.62), 'tier_3': (0.62, 0.42)}
    if tier in scores:
        return scores[tier][0] if tech else scores[tier][1]
    return 0.42 if tech else 0.28

def get_behavioral_mod(candidate):
    sig = candidate.get('redrob_signals', {})
    mod = 0.0
    if sig.get('open_to_work_flag'): mod += 0.08
    last_active = sig.get('last_active_date', '')
    if last_active:
        try:
            d = (datetime.now() - datetime.strptime(last_active, '%Y-%m-%d')).days
            if d < 30: mod += 0.08
            elif d > 180: mod -= 0.18
            elif d > 90: mod -= 0.08
        except: pass
    resp = sig.get('recruiter_response_rate', 0.5)
    if resp >= 0.60: mod += 0.06
    elif resp < 0.20: mod -= 0.12
    notice = sig.get('notice_period_days', 60)
    if notice <= 30: mod += 0.06
    elif notice > 90: mod -= 0.10
    elif notice > 60: mod -= 0.04
    return max(0.90, min(1.10, 1.0 + mod))

# ---------------------------------------------------------------------------
# Candidate analysis dump
# ---------------------------------------------------------------------------
def dump_top_features(scored, n=20):
    print(f"\n{'ID':<16} {'TITLE':<34} {'SCORE':<7} {'TITLE':<7} {'RETR':<6} {'RANK':<6} {'EVAL':<6} {'PROD':<6} {'CPROD':<6} {'YOE':<6} {'LOC':<6} {'EDU':<6} {'BEH':<6}")
    print('-' * 130)
    for e in scored[:n]:
        c = e['candidate']; p = c['profile']
        print(f"{e['candidate_id']:<16} {p['current_title'][:33]:<34} {e['score']:<7.4f} "
              f"{e['score_a']:<7.3f} {e['retrieval']:<6.3f} {e['ranking']:<6.3f} {e['eval_score']:<6.3f} {e['prod_score']:<6.3f} "
              f"{e['score_c']:<6.3f} {e['score_yoe']:<6.3f} {e['score_d']:<6.3f} {e['score_e']:<6.3f} {e['behavioral_mod']:<6.3f}")

# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------
def detect_honeypots(candidate):
    rules = set()
    skills = candidate.get('skills', [])
    ch = candidate.get('career_history', [])
    p = candidate.get('profile', {})
    sig = candidate.get('redrob_signals', {})
    edu = candidate.get('education', [])

    for s in skills:
        if s.get('proficiency') in ('advanced', 'expert') and s.get('duration_months', -1) == 0:
            rules.add('HP-1')
    total_career_y = sum(r.get('duration_months', 0) for r in ch) / 12
    if total_career_y > p.get('years_of_experience', 0) + 3:
        rules.add('HP-2')
    expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('endorsements', 0) == 0)
    if expert_zero >= 5: rules.add('HP-4')
    if len([r for r in ch if r.get('is_current', False)]) > 1: rules.add('HP-5')
    if edu:
        latest_end = max(e.get('end_year', 0) for e in edu)
        if latest_end >= datetime.now().year - 1 and latest_end + p.get('years_of_experience', 0) > datetime.now().year + 2: rules.add('HP-6')
    for score in sig.get('skill_assessment_scores', {}).values():
        if score == 100 and sig.get('profile_completeness_score', 100) < 30: rules.add('HP-7'); break
    return rules

# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------
def gen_reasoning(c, sa, sb, sc, sd, se, sf, bm, title_base, sr=None, srk=None, sev=None, sp=None):
    p = c['profile']
    sig = c.get('redrob_signals', {})
    skills = c.get('skills', [])
    career = c.get('career_history', [])
    title = p.get('current_title', 'N/A')
    yoe = p.get('years_of_experience', 0)
    location = p.get('location', '')
    country = p.get('country', '')

    parts = [f"{title} with {yoe}yrs"]

    # Identify specific evidence
    skill_names = [s['name'] for s in skills]
    retrieval_skills = [s for s in skill_names if s in RETRIEVAL_SKILLS | VECTOR_DB_SKILLS]
    ml_skills = [s for s in skill_names if s in PROD_ML_SKILLS]
    ranking_skills = [s for s in skill_names if s in RANKING_SKILLS]

    has_ranking_desc = srk is not None and srk >= 0.20
    has_eval_desc = sev is not None and sev >= 0.20

    # Build evidence strings
    evidence = []

    if has_ranking_desc or ranking_skills:
        evidence.append("ranking/recommendation experience")

    if retrieval_skills and not ranking_skills:
        top_r = retrieval_skills[:2]
        evidence.append(f"retrieval evidence ({', '.join(top_r)})")

    if ml_skills and not evidence:
        top_ml = ml_skills[:2]
        evidence.append(f"ML skills ({', '.join(top_ml)})")

    # Professional background
    prod_roles = [r for r in career
        if r.get('duration_months', 0) > 12
        and not any(s in r.get('company', '') for s in SERVICES_FIRMS)]
    if prod_roles:
        best = prod_roles[0]
        evidence.append(f"product experience at {best['company']}")

    if evidence:
        parts.append('; '.join(evidence[:2]) + ';')

    # Concerns
    concerns = []
    n = sig.get('notice_period_days', 60)
    if n > 60: concerns.append(f"{n}-day notice")
    if country != 'India' and not sig.get('willing_to_relocate'):
        concerns.append("outside India, not relocating")
    active = sig.get('last_active_date', '')
    if active:
        try:
            d = (datetime.now() - datetime.strptime(active, '%Y-%m-%d')).days
            if d > 90: concerns.append(f"inactive {d}d")
        except: pass
    if sb < 0.2: concerns.append("no retrieval/ML evidence")
    if title_base < 0.20: concerns.append("non-engineering background")
    parts.append(f"concern: {'; '.join(concerns[:2])}." if concerns else "well-rounded profile.")

    return ' '.join(parts).rstrip('.')

# ---------------------------------------------------------------------------
# Main ranking
# ---------------------------------------------------------------------------
def rank_candidates(candidates, semantic_scores=None):
    results = []
    for i, c in enumerate(candidates):
        if detect_honeypots(c):
            continue

        sa, title_base = score_title_relevance(c)
        sr, srk, sev, sp, sb = score_skills_evidence(c)
        sc = score_product_experience(c)
        sd = score_location(c)
        se = score_education(c)
        sy = score_yoe(c)

        raw = 0.30 * sa + 0.30 * sb + 0.20 * sc + 0.08 * sy + 0.07 * sd + 0.05 * se

        sf = semantic_scores[i] if semantic_scores is not None else 0.0
        raw = 0.90 * raw + 0.10 * sf

        bm = get_behavioral_mod(c)
        final = min(1.0, max(0.0, raw * bm))

        if title_base >= 0.50: pass
        elif title_base >= 0.20: final = min(final, 0.50)
        else: final = min(final, 0.30)

        tier = 'AI' if title_base >= 0.80 else ('DATA' if title_base >= 0.40 else ('ENG' if title_base >= 0.20 else 'NON'))
        results.append({
            'candidate_id': c['candidate_id'], 'score': final, 'score_a': sa,
            'retrieval': sr, 'ranking': srk, 'eval_score': sev, 'prod_score': sp,
            'score_b': sb, 'score_c': sc, 'score_yoe': sy, 'score_d': sd, 'score_e': se,
            'score_f': sf, 'behavioral_mod': bm, 'tier': tier,
            'title_base': title_base, 'candidate': c
        })

    results.sort(key=lambda x: (-round(x['score'], 6), x['candidate_id']))
    top = results[:100]

    out = []
    for i, r in enumerate(top):
        out.append({
            'candidate_id': r['candidate_id'], 'rank': i + 1,
            'score': round(r['score'], 6),
            'reasoning': gen_reasoning(r['candidate'], r['score_a'], r['score_b'],
                r['score_c'], r['score_d'], r['score_e'], r.get('score_f', 0.0),
                r['behavioral_mod'], r['title_base'],
                sr=r.get('retrieval'), srk=r.get('ranking'),
                sev=r.get('eval_score'), sp=r.get('prod_score'))
        })
    return out, results

# ---------------------------------------------------------------------------
# Build TF-IDF profile text
# ---------------------------------------------------------------------------
def build_profile_text(c):
    p = c.get('profile', {})
    parts = [p.get('current_title', ''), p.get('headline', ''), p.get('summary', '')]
    parts.append(' '.join(s.get('name', '') for s in c.get('skills', [])))
    for r in c.get('career_history', []):
        parts.append(f"{r.get('title', '')} at {r.get('company', '')}: {r.get('description', '')}")
    for e in c.get('education', []):
        parts.append(f"{e.get('field_of_study', '')} at {e.get('institution', '')}")
    return ' '.join(p for p in parts if p)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Redrob Ranker v5 (Data-Driven)')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-embeddings', action='store_true')
    parser.add_argument('--dump', action='store_true', help='Print feature dump of top candidates')
    args = parser.parse_args()

    with open(args.candidates) as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(candidates)} candidates")

    semantic_scores = None
    if not args.no_embeddings:
        print("Building TF-IDF vectorizer...")
        texts = [build_profile_text(c) for c in candidates]
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
            stop_words='english', sublinear_tf=True)
        all_t = [JD_TFIDF_TEXT] + texts
        t0 = datetime.now()
        tfidf = vec.fit_transform(all_t)
        semantic_scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
        print(f"  Done in {(datetime.now()-t0).total_seconds():.2f}s, "
              f"mean={semantic_scores.mean():.4f}, max={semantic_scores.max():.4f}")

    print("Ranking...")
    results, all_scored = rank_candidates(candidates, semantic_scores)
    if args.dump:
        dump_top_features(all_scored, 20)

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in results:
            w.writerow([r['candidate_id'], r['rank'], r['score'], r['reasoning']])
    print(f"Written {len(results)} candidates to {args.out}")

if __name__ == '__main__':
    main()
