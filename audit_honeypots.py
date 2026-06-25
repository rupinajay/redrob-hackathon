#!/usr/bin/env python3
"""
Audit honeypot detection rules from rank.py.
Loads 100K candidates, collects flagged & unflagged AI samples for manual inspection.
"""
import json, sys
from collections import defaultdict
from datetime import datetime
from copy import deepcopy

# ---------------------------------------------------------------------------
# Inline copy of detect_honeypots from rank.py  (lines 238-260)
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
    if expert_zero >= 5:
        rules.add('HP-4')
    if len([r for r in ch if r.get('is_current', False)]) > 1:
        rules.add('HP-5')
    if edu:
        latest_end = max(e.get('end_year', 2020) for e in edu)
        if latest_end + p.get('years_of_experience', 0) > datetime.now().year + 2:
            rules.add('HP-6')
    for score in sig.get('skill_assessment_scores', {}).values():
        if score == 100 and sig.get('profile_completeness_score', 100) < 30:
            rules.add('HP-7')
            break
    return rules

# ---------------------------------------------------------------------------
# Helpers for detailed rule evidence
# ---------------------------------------------------------------------------
def honeypot_evidence(candidate):
    """Return dict with rule -> (triggered, [evidence strings])"""
    ev = defaultdict(list)
    skills = candidate.get('skills', [])
    ch = candidate.get('career_history', [])
    p = candidate.get('profile', {})
    sig = candidate.get('redrob_signals', {})
    edu = candidate.get('education', [])

    # HP-1
    for s in skills:
        if s.get('proficiency') in ('advanced', 'expert') and s.get('duration_months', -1) == 0:
            ev['HP-1'].append(f"skill={s['name']} prof={s['proficiency']} dur=0")
    # HP-2
    total_career_y = sum(r.get('duration_months', 0) for r in ch) / 12
    yoe = p.get('years_of_experience', 0)
    if total_career_y > yoe + 3:
        ev['HP-2'].append(f"career_dur_y={total_career_y:.1f} yoe={yoe} threshold={yoe+3}")
    # HP-4
    expert_zero = [(s['name'], s.get('endorsements',0)) for s in skills
                   if s.get('proficiency') == 'expert' and s.get('endorsements', 0) == 0]
    if len(expert_zero) >= 5:
        ev['HP-4'].append(f"count={len(expert_zero)} skills={[x[0] for x in expert_zero]}")
    # HP-5
    current_roles = [r for r in ch if r.get('is_current', False)]
    if len(current_roles) > 1:
        ev['HP-5'].append(f"count={len(current_roles)} roles={[(r['title'],r['company']) for r in current_roles]}")
    # HP-6
    if edu:
        latest_end = max(e.get('end_year', 2020) for e in edu)
        if latest_end + yoe > datetime.now().year + 2:
            ev['HP-6'].append(f"latest_edu_end={latest_end} yoe={yoe} sum={latest_end+yoe} threshold={datetime.now().year+2}")
    # HP-7
    for k, score in sig.get('skill_assessment_scores', {}).items():
        if score == 100 and sig.get('profile_completeness_score', 100) < 30:
            ev['HP-7'].append(f"assessment={k}=100 completeness_score={sig.get('profile_completeness_score')}")
            break
    return dict(ev)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    src = '/Users/rupinajay/Developer/redrob/candidates.jsonl'
    max_candidates = 100_000

    flagged = []      # list of (candidate, rules_fired, evidence)
    unflagged_ai = [] # list of candidate

    AI_KW = ['ai', 'ml', 'nlp', 'search', 'recommendation', 'applied scientist']

    with open(src) as f:
        for i, line in enumerate(f):
            if i >= max_candidates:
                break
            if i % 10_000 == 0 and i > 0:
                print(f"  Progress: {i} lines, flagged={len(flagged)}, unflagged_ai={len(unflagged_ai)}", file=sys.stderr)

            line = line.strip()
            if not line:
                continue
            c = json.loads(line)

            rules = detect_honeypots(c)

            if rules:
                if len(flagged) < 50:
                    ev = honeypot_evidence(c)
                    flagged.append((c, rules, ev))
            else:
                if len(unflagged_ai) < 50:
                    title = c.get('profile', {}).get('current_title', '').lower()
                    if any(kw in title for kw in AI_KW):
                        unflagged_ai.append(c)

    print(f"\nDone. flagged={len(flagged)}  unflagged_ai={len(unflagged_ai)}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Dump flagged
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("SECTION 1: FLAGGED HONEYPOTS (first 50)")
    print("=" * 100)

    for idx, (c, rules, ev) in enumerate(flagged, 1):
        p = c.get('profile', {})
        ch = c.get('career_history', [])
        edu = c.get('education', [])
        skills = c.get('skills', [])

        print(f"\n{'─'*100}")
        print(f"FLAGGED #{idx}  |  ID: {c['candidate_id']}")
        print(f"{'─'*100}")
        print(f"Title:          {p.get('current_title', 'N/A')}")
        print(f"YOE:            {p.get('years_of_experience', 0)}")
        print(f"Company:        {p.get('current_company', 'N/A')}")
        print(f"Location:       {p.get('location', 'N/A')}, {p.get('country', 'N/A')}")
        print(f"Rules triggered: {', '.join(sorted(rules))}")
        print()
        for rule in sorted(ev.keys()):
            for detail in ev[rule]:
                print(f"  [EVIDENCE {rule}] {detail}")

        print(f"\n  Skills ({len(skills)} total):")
        for s in skills[:15]:
            print(f"    - {s['name']:30s} prof={s['proficiency']:12s} dur={s.get('duration_months',0):3d}mo  end={s.get('endorsements',0)}")
        if len(skills) > 15:
            print(f"    ... and {len(skills) - 15} more")

        print(f"\n  Career History:")
        for r in ch:
            curr = " [CURRENT]" if r.get('is_current') else ""
            print(f"    - {r['title']:35s} @ {r['company']:25s} {r.get('duration_months',0):3d}mo{curr}")

        print(f"\n  Education:")
        for e in edu:
            print(f"    - {e.get('field_of_study','N/A'):30s} {e.get('degree','N/A'):10s} end={e.get('end_year','N/A')}  {e.get('institution','N/A')}")

    # -----------------------------------------------------------------------
    # Dump unflagged AI
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("SECTION 2: UNFLAGGED AI/ML CANDIDATES (50 samples)")
    print("=" * 100)

    for idx, c in enumerate(unflagged_ai, 1):
        p = c.get('profile', {})
        ch = c.get('career_history', [])
        edu = c.get('education', [])
        skills = c.get('skills', [])

        print(f"\n{'─'*100}")
        print(f"UNFLAGGED AI #{idx}  |  ID: {c['candidate_id']}")
        print(f"{'─'*100}")
        print(f"Title:          {p.get('current_title', 'N/A')}")
        print(f"YOE:            {p.get('years_of_experience', 0)}")
        print(f"Company:        {p.get('current_company', 'N/A')}")
        print(f"Location:       {p.get('location', 'N/A')}, {p.get('country', 'N/A')}")

        print(f"\n  Skills ({len(skills)} total):")
        for s in skills[:15]:
            print(f"    - {s['name']:30s} prof={s['proficiency']:12s} dur={s.get('duration_months',0):3d}mo  end={s.get('endorsements',0)}")
        if len(skills) > 15:
            print(f"    ... and {len(skills) - 15} more")

        print(f"\n  Career History:")
        for r in ch:
            curr = " [CURRENT]" if r.get('is_current') else ""
            print(f"    - {r['title']:35s} @ {r['company']:25s} {r.get('duration_months',0):3d}mo{curr}")

        print(f"\n  Education:")
        for e in edu:
            print(f"    - {e.get('field_of_study','N/A'):30s} {e.get('degree','N/A'):10s} end={e.get('end_year','N/A')}  {e.get('institution','N/A')}")

if __name__ == '__main__':
    main()
