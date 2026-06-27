#!/usr/bin/env python3
"""
Redrob Hackathon Ranker v6 — PRP v2.0 Compliant

Architecture:
  1. Honeypot pre-filter (7 rules)
  2. Title gate with seniority + career modifiers
  3. PRP-corrected weights: A=35%, B=25%, C=18%, D=12%, E=10%
  4. B = average of 4 sub-scores (B1 embeddings, B2 vector DB, B3 prod ML, B4 eval)
  5. C = YOE band + AI/ML proportion bonus + product proportion
  6. Behavioral [0.40, 1.20] with 11 PRP signals
  7. Optional TF-IDF (--with-embeddings, 10% blend)
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
# Retrieval evidence — split into B1 (embeddings) and B2 (vector DB) per PRP
EMBEDDING_SKILLS = {'sentence-transformers', 'BGE', 'E5', 'dense retrieval', 'bi-encoder',
    'sparse retrieval', 'hybrid search', 'embedding'}
FAISS_SKILLS = {'FAISS', 'Annoy'}
VECTOR_DB_SKILLS = {'Pinecone', 'Qdrant', 'Weaviate', 'Milvus', 'Chroma', 'Elasticsearch', 'OpenSearch'}
RANKING_SKILLS = {'NDCG', 'MRR', 'MAP', 'learning to rank', 'LambdaRank',
    'recommendation systems', 'recommender', 'matching', 're-ranking'}
EVAL_SKILLS = {'A/B testing', 'offline evaluation', 'ranking evaluation', 'evaluation', 'metrics'}
PROD_ML_SKILLS = {'PyTorch', 'TensorFlow', 'scikit-learn', 'Hugging Face', 'transformers',
    'Keras', 'Langchain', 'FastAPI', 'Flask', 'Docker', 'Kubernetes', 'CI/CD',
    'MLOps', 'Airflow', 'Kubeflow', 'GCP', 'AWS', 'Azure', 'Spark', 'Python'}

EMBEDDING_DESC_KW = ['semantic search', 'dense retrieval', 'embedding index', 'bi-encoder', 'cross-encoder']
VECTOR_DB_DESC_KW = ['vector database', 'ann index', 'hybrid search', 'sparse+dense', 'vector search']
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
    title_score = base * mult

    # Career trajectory modifiers per PRP
    career = candidate.get('career_history', [])
    mods = 0.0
    if career:
        avg_tenure = sum(r.get('duration_months', 0) for r in career) / len(career)
        if avg_tenure >= 18: mods += 0.05
    if len(career) >= 2:
        titles = [r.get('title', '').lower() for r in career]
        if any(k in titles[-1] for k in SENIORITY_KW):
            if not any(k in titles[0] for k in SENIORITY_KW):
                mods += 0.05
    desc = ' '.join(r.get('description', '') for r in career).lower()
    if any(k in desc for k in ['built', 'designed', 'shipped', 'deployed', 'owned', 'led', 'architected']):
        mods += 0.05

    return min(1.0, title_score + mods), base

def score_skills_evidence(candidate):
    """Return (B1_embeddings, B2_vector_db, B4_eval, B3_prod_python, composite) per PRP."""
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

    # B1: Embeddings-based retrieval
    b1 = max(best(EMBEDDING_SKILLS | FAISS_SKILLS, 0.85), desc_density(EMBEDDING_DESC_KW, 0.35), title_density(EMBEDDING_DESC_KW, 0.25))
    if b1 == 0.0 and best(VECTOR_DB_SKILLS, 0.50) > 0:
        b1 = 0.10

    # B2: Vector DB / hybrid search
    b2 = max(best(VECTOR_DB_SKILLS, 0.85), desc_density(VECTOR_DB_DESC_KW, 0.35), title_density(VECTOR_DB_DESC_KW, 0.25))
    if b2 == 0.0 and best(FAISS_SKILLS, 0.50) > 0:
        b2 = 0.10

    # B3: Production ML + Python
    b3 = max(best(PROD_ML_SKILLS, 0.70), desc_density(PROD_DESC_KW, 0.25))
    if 'Python' in [s['name'] for s in skills]:
        py_skill = [s for s in skills if s['name'] == 'Python'][0]
        b3 = max(b3, min(1.0, 0.85 * skill_trust(py_skill)))

    # B4: Ranking evaluation frameworks
    b4 = max(best(EVAL_SKILLS, 0.80), desc_density(EVAL_DESC_KW, 0.30))

    # Composite = simple average per PRP spec
    composite = (b1 + b2 + b3 + b4) / 4.0
    return b1, b2, b3, b4, composite

def score_experience(candidate):
    """Component C: YOE band + product proportion + AI/ML role proportion."""
    career = candidate.get('career_history', [])
    yoe = candidate.get('profile', {}).get('years_of_experience', 0)

    # YOE band
    if 5 <= yoe <= 9: yoe_score = 1.00
    elif 4 <= yoe < 5 or 9 < yoe <= 11: yoe_score = 0.82
    elif 3 <= yoe < 4 or 11 < yoe <= 14: yoe_score = 0.58
    else: yoe_score = 0.25

    # Product proportion
    prop = get_product_prop(career)
    all_services = all(any(s in r.get('company', '') for s in SERVICES_FIRMS) for r in career) if career else False

    # AI/ML role proportion bonus per PRP
    ai_core = {'ml engineer', 'ai engineer', 'machine learning engineer', 'ai/ml engineer',
        'ai research engineer', 'applied scientist', 'applied ml engineer', 'ai specialist',
        'search engineer', 'recommendation engineer', 'recommendation systems engineer',
        'ranking engineer', 'matching engineer', 'nlp engineer', 'computer vision engineer'}
    ai_months = sum(r.get('duration_months', 0) for r in career
                    if any(ai in r.get('title', '').lower() for ai in ai_core))
    total_months = sum(r.get('duration_months', 0) for r in career)
    ai_bonus = (ai_months / total_months * 0.20) if total_months > 0 else 0.0
    prod_bonus = prop * 0.15

    raw = yoe_score + prod_bonus + ai_bonus
    if all_services: raw -= 0.20
    return min(1.0, max(0.0, raw))

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
    if sig.get('open_to_work_flag'): mod += 0.10
    last_active = sig.get('last_active_date', '')
    if last_active:
        try:
            d = (datetime.now() - datetime.strptime(last_active, '%Y-%m-%d')).days
            if d < 30: mod += 0.12
            elif 90 <= d <= 180: mod -= 0.12
            elif d > 180: mod -= 0.28
        except: pass
    resp = sig.get('recruiter_response_rate', 0.5)
    if resp >= 0.60: mod += 0.08
    elif resp < 0.20: mod -= 0.18
    icr = sig.get('interview_completion_rate', 0.5)
    if icr >= 0.80: mod += 0.06
    elif icr < 0.40: mod -= 0.12
    notice = sig.get('notice_period_days', 60)
    if notice <= 30: mod += 0.08
    elif 61 <= notice <= 90: mod -= 0.06
    elif notice > 90: mod -= 0.14
    gh = sig.get('github_activity_score', 0)
    if gh > 50: mod += 0.06
    art = sig.get('avg_response_time_hours', 999)
    if art < 12: mod += 0.04
    elif art > 72: mod -= 0.06
    if sig.get('verified_email') and sig.get('verified_phone'): mod += 0.03
    return max(0.40, min(1.20, 1.0 + mod))

# ---------------------------------------------------------------------------
# Candidate analysis dump
# ---------------------------------------------------------------------------
def dump_top_features(scored, n=20):
    print(f"\n{'ID':<16} {'TITLE':<34} {'SCORE':<7} {'A_TTL':<6} {'B1_EMB':<7} {'B2_VDB':<7} {'B3_ML':<7} {'B4_EVAL':<8} {'C_EXP':<6} {'D_LOC':<6} {'E_EDU':<6} {'F_BEH':<6}")
    print('-' * 130)
    for e in scored[:n]:
        c = e['candidate']; p = c['profile']
        print(f"{e['candidate_id']:<16} {p['current_title'][:33]:<34} {e['score']:<7.4f} "
              f"{e['score_a']:<6.3f} {e['b1']:<7.3f} {e['b2']:<7.3f} {e['b3']:<7.3f} {e['b4']:<8.3f} "
              f"{e['score_c']:<6.3f} {e['score_d']:<6.3f} {e['score_e']:<6.3f} {e['behavioral_mod']:<6.3f}")

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
    for r in ch:
        start = r.get('start_date', '')
        if start and r.get('is_current', False):
            try:
                years = (datetime.now() - datetime.strptime(start, '%Y-%m-%d')).days / 365.25
                if years > 12:
                    co = r.get('company', '').lower()
                    if any(k in co for k in {'ai', 'tech', 'labs', 'data', 'digital', 'soft', 'solution', 'app'}):
                        rules.add('HP-3'); break
            except: pass
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
def gen_reasoning(c, sa, sb, sc, sd, se, sf, bm, title_base, b1=None, b2=None, b3=None, b4=None):
    p = c['profile']
    sig = c.get('redrob_signals', {})
    skills = c.get('skills', [])
    career = c.get('career_history', [])
    title = p.get('current_title', 'N/A')
    yoe = p.get('years_of_experience', 0)
    location = p.get('location', '')
    country = p.get('country', '')

    parts = [f"{title} with {yoe}yrs"]

    skill_names = [s['name'] for s in skills]
    evidence = []

    has_ranking_skill = any(s in RANKING_SKILLS for s in skill_names)
    has_eval_skill = any(s in EVAL_SKILLS for s in skill_names)
    has_vector_db = any(s in VECTOR_DB_SKILLS for s in skill_names)
    has_embedding = any(s in EMBEDDING_SKILLS | FAISS_SKILLS for s in skill_names)

    if has_ranking_skill or (b4 is not None and b4 >= 0.20):
        evidence.append("ranking/recommendation experience")

    if has_vector_db:
        top_vdb = [s for s in skill_names if s in VECTOR_DB_SKILLS][:2]
        evidence.append(f"vector db ({', '.join(top_vdb)})")
    elif has_embedding:
        top_emb = [s for s in skill_names if s in EMBEDDING_SKILLS | FAISS_SKILLS][:2]
        evidence.append(f"embeddings ({', '.join(top_emb)})")

    prod_roles = [r for r in career
        if r.get('duration_months', 0) > 12
        and not any(s in r.get('company', '') for s in SERVICES_FIRMS)]
    if prod_roles:
        evidence.append(f"product exp at {prod_roles[0]['company']}")

    if evidence:
        parts.append('; '.join(evidence[:2]) + ';')

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
        b1, b2, b3, b4, sb = score_skills_evidence(c)
        sc = score_experience(c)
        sd = score_location(c)
        se = score_education(c)

        raw = 0.35 * sa + 0.25 * sb + 0.18 * sc + 0.12 * sd + 0.10 * se

        sf = semantic_scores[i] if semantic_scores is not None else 0.0
        if sf > 0:
            raw = 0.90 * raw + 0.10 * sf

        bm = get_behavioral_mod(c)
        final = min(1.0, max(0.0, raw * bm))

        if title_base >= 0.50: pass
        elif title_base >= 0.20: final = min(final, 0.50)
        else: final = min(final, 0.30)

        results.append({
            'candidate_id': c['candidate_id'], 'score': final, 'score_a': sa,
            'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4,
            'score_b': sb, 'score_c': sc, 'score_d': sd, 'score_e': se,
            'score_f': sf, 'behavioral_mod': bm,
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
                b1=r.get('b1'), b2=r.get('b2'), b3=r.get('b3'), b4=r.get('b4'))
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
    parser = argparse.ArgumentParser(description='Redrob Ranker v6 (PRP-Compliant)')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--with-embeddings', action='store_true', help='Enable TF-IDF semantic similarity (10 pct blend, not in PRP spec)')
    parser.add_argument('--dump', action='store_true', help='Print feature dump of top candidates')
    args = parser.parse_args()

    with open(args.candidates) as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(candidates)} candidates")

    semantic_scores = None
    if args.with_embeddings:
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
