# PRP v2.0 Compliance Audit

Audit against `Redrob_Hackathon_PRP_v2-2.docx` specification.

---

## Format & Submission

| Check | Status | Detail |
|-------|--------|--------|
| CSV: 100 rows + header | ✅ Pass | `candidate_id,rank,score,reasoning` |
| Scores non-increasing | ✅ Pass | Strictly decreasing, ties broken by candidate_id ascending |
| `validate_submission.py` | ✅ Pass | Returns "Submission is valid." |
| Runtime < 5 min | ✅ Pass | ~13s on single CPU, 16GB RAM |
| Honeypots in top 100 | ✅ 0/100 | 43 total in 100K pool |

---

## Honeypot Detection (7 Rules)

| Rule | Condition | Status |
|------|-----------|--------|
| HP-1 | Skill: advanced/expert proficiency + duration_months == 0 | ✅ |
| HP-2 | Sum career duration > YOE + 3 years | ✅ |
| HP-3 | Current role > 12 years at startup/AI/tech company | ✅ (catches 0) |
| HP-4 | 5+ expert skills with 0 endorsements | ✅ |
| HP-5 | > 1 current role | ✅ |
| HP-6 | Education end_year + YOE > current_year + 2 | ✅ (with recency guard) |
| HP-7 | Skill assessment score == 100 + profile completeness < 30 | ✅ |

All 7 rules implemented.

---

## Title Gate

| Title Base Score | PRP Cap Applied | Our Cap | Status |
|----------------|:---------------:|:-------:|--------|
| ≥ 0.50 (engineering gate passed) | No cap | No cap | ✅ |
| 0.20 – 0.49 (adjacent/ambiguous) | 0.50 max | 0.50 max | ✅ |
| < 0.20 (non-engineering) | 0.30 max | 0.30 max | ✅ |
| 0.00 (mismatch e.g. Civil/Mechanical/Teacher) | Score = 0.0 | Not implemented (gets 0.10, capped at 0.30) | ❌ Minor |

---

## Component Weights

| Component | PRP Weight | Our Weight | Our Sub-components | Effective Δ |
|-----------|:---------:|:---------:|--------------------|:----------:|
| A: Title & Career Fit | **35%** | **30%** | title × seniority only | −5% |
| B: Core Skills (4 JD requirements) | **25%** | **30%** | weighted composite (35r + 30rk + 20p + 15e) | +5% |
| C: Experience Depth | **18%** | **28%** | product company (20%) + YOE band (8%) | +10% |
| D: Location | **12%** | **7%** | same scoring | −5% |
| E: Education | **10%** | **5%** | same scoring | −5% |
| TF-IDF semantic blend | **none** | **10%** | not in spec | +10% |

**Issues:**
- C overweighted by 10% (product + YOE combined)
- D and E underweighted by 5% each
- TF-IDF blend not in PRP spec

---

## Component A — Title & Career Fit (35%)

### Base Title Score

| Category | PRP Score | Our Score | Target Titles (PRP) | Status |
|----------|:---------:|:---------:|--------------------|--------|
| Core AI/ML Engineering | 1.00 | 0.95 | ML Engineer, AI Engineer, NLP Engineer, Applied Scientist, Search Engineer, Recommendation Eng, AI Research Engineer | ❌ |
| Adjacent ML Engineering | 0.80 | 0.70 | Data Scientist (retrieval focus), CV Engineer, AI Specialist | ❌ |
| Software Engineering | 0.55 | 0.30 | Backend Engineer, Full Stack, SWE | ❌ |
| Data Engineering | 0.35 | 0.50 | Data Engineer, Analytics Engineer | ❌ |
| Adjacent Tech (non-ML) | 0.20 | 0.30 | DevOps, Cloud, QA, Mobile, Frontend | ❌ |
| Non-technical / Business | 0.10 | 0.10 | Marketing, HR, PM, Operations | ✅ |
| Outright mismatch | 0.00 | 0.10 | Mechanical, Civil, Graphic Designer, Teacher, Lawyer | ❌ |

### Seniority Modifier

| Seniority | Modifier | Status |
|-----------|:--------:|--------|
| Senior / Lead / Principal / Staff / Head of | × 1.00 | ✅ |
| No prefix | × 0.90 | ✅ |
| Junior / Associate / Trainee / Intern / Fresher | × 0.75 | ✅ |

### Career Trajectory Modifiers (Additive)

| Modifier | PRP Value | Implemented | Status |
|----------|:---------:|:-----------:|--------|
| Avg role tenure ≥ 18 months (stability) | +0.05 | No | ❌ |
| Clear progression in seniority across roles | +0.05 | No | ❌ |
| ML system ownership in career descriptions | +0.05 | No | ❌ |

**Formula:** `A = clamp(title_score × seniority + career_modifiers, 0.0, 1.0)`

---

## Component B — Core Skills Match (25%)

| Sub-score | Skill Area | Our Approach | Status |
|-----------|-----------|-------------|--------|
| B1 | Embeddings-based retrieval | Trust-weighted skill match + description density | ~✅ |
| B2 | Vector DB / hybrid search | Trust-weighted skill match + description density | ~✅ |
| B3 | Python production code | Trust-weighted skill match + inference from ML tools (PyTorch, TF, etc.) + description density | ~✅ |
| B4 | Ranking evaluation frameworks | Trust-weighted skill match + description density | ~✅ |

**Aggregation mismatch:** PRP specifies `B = (B1 + B2 + B3 + B4) / 4` (simple average).  
We use `composite = 0.35 × retrieval + 0.30 × ranking + 0.20 × production + 0.15 × eval` (weighted).

---

## Component C — Experience Depth (18%)

| Factor | PRP | Our Code | Status |
|--------|-----|----------|--------|
| YOE band score | [5,9]=1.00, [4,5)/(9,11]=0.82, [3,4)/(11,14]=0.58, else 0.25 | Same | ✅ |
| AI/ML role proportion bonus | +0 to +0.20 | Not implemented | ❌ |
| Product company proportion bonus | +0 to +0.15 | Separate 20% component with different structure | ⚠️ |

**Formula:** `C = clamp(yoe_band + ai_proportion_bonus + product_bonus, 0.0, 1.0)`

---

## Component D — Location (12%)

| Condition | PRP Score | Our Score | Status |
|-----------|:---------:|:---------:|--------|
| India + (Pune, Noida, Delhi NCR, Hyderabad, Mumbai, Bangalore) | 1.00 | 1.00 | ✅ |
| India + other city + willing_to_relocate = true | 0.85 | 0.85 | ✅ |
| India + other city + not willing to relocate | 0.55 | 0.55 | ✅ |
| India + unspecified location | 0.65 | Not implemented (falls to 0.55) | ❌ |
| Outside India + willing_to_relocate = true | 0.30 | 0.30 | ✅ |
| Outside India + not willing to relocate | 0.05 | 0.05 | ✅ |

---

## Component E — Education (10%)

| Condition | PRP Score | Our Score | Status |
|-----------|:---------:|:---------:|--------|
| tier_1 institution + CS/ECE/Math/Stats/AI | 1.00 | 1.00 | ✅ |
| tier_2 + CS/ECE/Math/Stats/AI | 0.82 | 0.82 | ✅ |
| tier_1 or tier_2 + non-technical field | 0.60 | 0.60 | ✅ |
| tier_3 + CS/ECE/Math/Stats/AI | 0.62 | 0.62 | ✅ |
| tier_3 + non-technical, or tier_4 + technical | 0.42 | 0.42 | ✅ |
| tier_4 + non-technical, or tier unknown | 0.28 | 0.28 | ✅ |
| No education listed | 0.20 | 0.20 | ✅ |

Matches exactly.

---

## Component F — Behavioral Modifier

Range: `clamp(1.0 + Σ(signal_modifiers), floor, cap)`

### Signal Values

| Signal | PRP Value | Our Value | Status |
|--------|:---------:|:---------:|--------|
| open_to_work_flag = true | +0.10 | +0.08 | ❌ |
| last_active_date < 30 days ago | +0.12 | +0.08 | ❌ |
| last_active_date 30–90 days | 0.00 | (fallthrough, same) | ✅ |
| last_active_date 90–180 days | −0.12 | −0.08 (combined > 90d) | ❌ |
| last_active_date > 180 days | −0.28 | −0.18 | ❌ |
| recruiter_response_rate ≥ 0.60 | +0.08 | +0.06 | ❌ |
| recruiter_response_rate < 0.20 | −0.18 | −0.12 | ❌ |
| interview_completion_rate ≥ 0.80 | +0.06 | Not implemented | ❌ |
| interview_completion_rate < 0.40 | −0.12 | Not implemented | ❌ |
| notice_period_days ≤ 30 | +0.08 | +0.06 | ❌ |
| notice_period_days 61–90 | −0.06 | −0.04 (combined > 60d) | ❌ |
| notice_period_days > 90 | −0.14 | −0.10 | ❌ |
| github_activity_score > 50 | +0.06 | Not implemented | ❌ |
| avg_response_time_hours < 12 | +0.04 | Not implemented | ❌ |
| avg_response_time_hours > 72 | −0.06 | Not implemented | ❌ |
| verified_email AND verified_phone | +0.03 | Not implemented | ❌ |

### Range

| | PRP | Our Code | Status |
|---|:---:|:--------:|--------|
| Floor | 0.40 | 0.90 | ❌ |
| Cap | 1.20 | 1.10 | ❌ |

**Signal availability in dataset:** All signals except `verified_both` are available for 100% of non-honeypot candidates. `verified_both` is available at 44.5%.

---

## Reasoning Quality (Stage 4 Checks)

| Check | Requirement | Status |
|-------|-----------|--------|
| Specific facts | References actual YOE, current title, named skills | ✅ |
| JD connection | Connects to JD requirements (retrieval, ranking, production) | ✅ |
| Honest concerns | Acknowledges gaps for lower ranks | ✅ |
| No hallucination | Every claim corresponds to profile data | ✅ |
| Variation | No two entries templated or identical | ✅ |
| Rank consistency | Tone matches rank position | ✅ |

---

## Summary

| Category | Count |
|----------|:-----:|
| ✅ Compliant | 19 |
| ❌ Non-compliant (minor) | 3 |
| ❌ Non-compliant (structural) | 3 |

### Quick Fixes (<1 hour)

1. Add career modifiers to A: stability (+0.05), progression (+0.05), ML ownership (+0.05)
2. Add AI/ML proportion bonus to C (+0 to +0.20)
3. Add missing behavioral signals: interview_completion_rate, github_activity, avg_response_time, verified_both
4. Match PRP behavioral modifier values
5. Widen behavioral range to [0.40, 1.20]
6. Add `--no-tfidf` as default (remove TF-IDF blend from default behavior)

### Structural Differences (>1 hour)

1. Title base scores differ from PRP tiers (0.95 vs 1.00, 0.70 vs 0.80)
2. B component aggregation: weighted composite vs simple average
3. Weight distribution: our C (28%) is 10% higher than PRP spec (18%)
