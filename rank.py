#!/usr/bin/env python3
"""
Redrob Ranker — config-driven production ranking with multi-view semantic matching,
soft title gates, trajectory quality, and description-validated skills.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv [--config ./config.json] [--dump]
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    required = {"weights", "archetype", "skills", "career", "location",
                "education", "behavioral", "honeypots", "semantic", "services_firms"}
    missing = required - set(cfg.keys())
    if missing:
        sys.exit(f"Config missing required sections: {', '.join(sorted(missing))}")
    w = cfg["weights"]
    if abs(sum(w.values()) - 1.0) > 0.001:
        sys.exit(f"Config weights sum to {sum(w.values()):.3f}, must be 1.0")
    return cfg


# ---------------------------------------------------------------------------
# Skill trust — duration- and endorsement-adjusted
# ---------------------------------------------------------------------------
def skill_trust(skill, cfg):
    prof = skill.get("proficiency", cfg["skills"]["default_proficiency"])
    base = cfg["skills"]["proficiency_scores"].get(prof, 0.25)
    dur = skill.get("duration_months", 0)
    end = skill.get("endorsements", 0)
    max_dur = cfg["skills"]["max_trust_duration_months"]
    boost_factor = cfg["skills"]["endorsement_boost_factor"]
    if dur == 0 and prof in ("advanced", "expert"):
        return 0.0
    return base * min(1.0, dur / max_dur) * min(1.15, 1.0 + end / boost_factor)


# ---------------------------------------------------------------------------
# Archetype / title
# ---------------------------------------------------------------------------
def _match_tier(title_lower, arch):
    for tier in arch["tiers"]:
        for kw in tier["keywords"]:
            if kw in title_lower:
                return tier
    return None


def _seniority_mult(title_lower, arch):
    if any(k in title_lower for k in arch["seniority_keywords"]):
        return arch["seniority_multiplier"]
    if any(k in title_lower for k in arch["junior_keywords"]):
        return arch["junior_multiplier"]
    return arch["no_seniority_multiplier"]


def score_archetype(candidate, cfg):
    arch = cfg["archetype"]
    title = candidate.get("profile", {}).get("current_title", "").lower()
    tier = _match_tier(title, arch)
    if tier is None:
        return arch["default_score"], arch["default_score"], None

    base = tier["base_score"]
    mult = _seniority_mult(title, arch)
    score = base * mult

    career = candidate.get("career_history", [])
    if career:
        avg_tenure = sum(r.get("duration_months", 0) for r in career) / len(career)
        if avg_tenure >= arch["career_tenure_bonus_months"]:
            score += arch["career_tenure_bonus"]
    if len(career) >= 2:
        titles = [r.get("title", "").lower() for r in career]
        if any(k in titles[-1] for k in arch["seniority_keywords"]):
            if not any(k in titles[0] for k in arch["seniority_keywords"]):
                score += arch["career_progression_bonus"]
    desc = " ".join(r.get("description", "") for r in career).lower()
    if any(k in desc for k in arch["action_verbs"]):
        score += arch["action_verb_bonus"]

    capped = min(tier["cap"], score)
    return capped, base, tier["label"]


# ---------------------------------------------------------------------------
# Skills — description-validated, config-driven groups
# ---------------------------------------------------------------------------
def _best_skill_score(skill_names, group_skills, trust_mul, candidate, cfg):
    career = candidate.get("career_history", [])
    desc_text = " ".join(r.get("description", "") for r in career).lower()
    desc_boost = cfg["skills"].get("description_boost", 0.0)
    overrides = cfg["skills"].get("skill_overrides", {})

    best = 0.0
    for s in candidate.get("skills", []):
        name = s.get("name", "")
        if name in group_skills:
            mul = overrides.get(name, {}).get("trust_mul", trust_mul)
            base = min(1.0, mul * skill_trust(s, cfg))
            if name.lower() in desc_text:
                base = min(1.0, base + desc_boost)
            best = max(best, base)
    return best


def _desc_density(career, kws, max_score):
    mentions = 0
    for r in career:
        d = (r.get("description", "") or "").lower()
        mentions += sum(d.count(k) for k in kws)
    if mentions == 0:
        return 0.0
    return min(max_score, max_score * math.sqrt(mentions) / 3.0)


def _title_density(career_titles, kws, max_score):
    mentions = sum(career_titles.count(k) for k in kws)
    if mentions == 0:
        return 0.0
    return min(max_score, max_score * math.sqrt(mentions) / 3.0)


def score_skills_evidence(candidate, cfg):
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    desc = " ".join(r.get("description", "") for r in career).lower()
    career_titles = " ".join(r.get("title", "") for r in career).lower()
    skill_names = {s.get("name") for s in skills}
    groups = cfg["skills"]["groups"]
    scores = {}

    for gid, g in groups.items():
        g_skills = set(g["skills"])
        g_kws = g.get("desc_kw", [])
        g_title_kws = g.get("title_kw", g_kws)

        s1 = _best_skill_score(skill_names, g_skills, g["trust_mul"], candidate, cfg)
        s2 = _desc_density(career, g_kws, g["desc_max"]) if g_kws else 0.0
        s3 = _title_density(career_titles, g_title_kws, g["title_max"]) if g.get("title_max", 0) > 0 else 0.0
        score = max(s1, s2, s3)

        if score == 0.0 and g.get("fallback_group") and g["fallback_group"] in scores:
            if scores[g["fallback_group"]] > 0:
                score = g["fallback_score"]

        scores[gid] = score

    composite = sum(scores.values()) / max(len(scores), 1)
    return scores, composite


# ---------------------------------------------------------------------------
# Career — trajectory quality, switcher detection, YOE bands
# ---------------------------------------------------------------------------
def score_career_pattern(candidate, cfg):
    career_config = cfg["career"]
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    ch = candidate.get("career_history", [])
    total_months = sum(r.get("duration_months", 0) for r in ch)

    # YOE band
    yoe_score = career_config["default_yoe_score"]
    for band in career_config["yoe_bands"]:
        if band["min"] <= yoe <= band["max"]:
            yoe_score = band["score"]
            break

    # Product proportion
    prod_months = sum(
        r.get("duration_months", 0) for r in ch
        if not any(s in r.get("company", "") for s in cfg["services_firms"])
    )
    prod_prop = prod_months / total_months if total_months > 0 else 0.0

    # AI role proportion
    ai_title_kw = career_config["ai_title_keywords"]
    ai_months = sum(
        r.get("duration_months", 0) for r in ch
        if any(ai in r.get("title", "").lower() for ai in ai_title_kw)
    )
    ai_prop = ai_months / total_months if total_months > 0 else 0.0

    score = yoe_score
    score += prod_prop * career_config["product_bonus_weight"]
    score += ai_prop * career_config["ai_role_bonus_weight"]

    # All-services penalty
    all_services = (
        all(any(s in r.get("company", "") for s in cfg["services_firms"]) for r in ch)
        if ch else False
    )
    if all_services:
        score -= career_config["all_services_penalty"]

    # ---------------------------------------------------------------
    # Trajectory quality
    # ---------------------------------------------------------------
    traj = career_config.get("trajectory", {})
    promotions = 0
    # Group roles by company to detect promotions
    by_company = {}
    for r in ch:
        co = r.get("company", "")
        by_company.setdefault(co, []).append(r)
    for co, roles in by_company.items():
        if len(roles) < 2:
            continue
        sorted_roles = sorted(roles, key=lambda x: x.get("start_date", ""))
        for i in range(1, len(sorted_roles)):
            prev_t = sorted_roles[i - 1].get("title", "").lower()
            cur_t = sorted_roles[i].get("title", "").lower()
            prev_senior = any(k in prev_t for k in cfg["archetype"]["seniority_keywords"])
            cur_senior = any(k in cur_t for k in cfg["archetype"]["seniority_keywords"])
            prev_non_senior = not prev_senior
            if prev_non_senior and cur_senior:
                promotions += 1

    if promotions > 0:
        score += min(promotions * traj.get("promotion_bonus", 0.05), 0.10)

    # Stability: prefer roles 12-48 months (not job-hopper, not stagnant)
    stable = sum(1 for r in ch if traj.get("stability_band_min_months", 12) <= r.get("duration_months", 0) <= traj.get("stability_band_max_months", 48))
    if len(ch) > 0 and stable / len(ch) >= 0.5:
        score += traj.get("stability_bonus", 0.03)

    # Career switcher bonus: earliest role is non-AI, latest role is AI
    if len(ch) >= 2:
        sorted_ch = sorted(ch, key=lambda x: x.get("start_date", ""))
        first_title = sorted_ch[0].get("title", "").lower()
        last_title = sorted_ch[-1].get("title", "").lower()
        first_is_ai = any(ai in first_title for ai in ai_title_kw)
        last_is_ai = any(ai in last_title for ai in ai_title_kw)
        if not first_is_ai and last_is_ai:
            score += traj.get("switcher_bonus", 0.08)

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------
def score_location(candidate, cfg):
    loc_cfg = cfg["location"]
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    country = p.get("country", "")
    location = p.get("location", "").lower()
    relo = sig.get("willing_to_relocate", False)
    pref = any(c in location for c in loc_cfg["preferred_cities"])

    if country in loc_cfg.get("country_preference", {}):
        if pref:
            return loc_cfg.get("preferred_city_score", 1.0)
        return loc_cfg["no_preferred_city_relocate"] if relo else loc_cfg["no_preferred_city_no_relocate"]
    return loc_cfg["overseas_relocate"] if relo else loc_cfg["overseas_no_relocate"]


# ---------------------------------------------------------------------------
# Education — with research degree bonus
# ---------------------------------------------------------------------------
def score_education(candidate, cfg):
    edu_cfg = cfg["education"]
    edu = candidate.get("education", [])
    if not edu:
        return edu_cfg["default_score"]

    best = max(edu, key=lambda e: e.get("end_year", 0))
    tier = best.get("tier", "unknown")
    field = best.get("field_of_study", "").lower()
    degree = best.get("degree", "").lower()
    tech = any(f in field for f in edu_cfg["tech_fields"])

    if tier in edu_cfg["tiers"]:
        score = edu_cfg["tiers"][tier]["tech"] if tech else edu_cfg["tiers"][tier]["non_tech"]
    else:
        score = edu_cfg["default_tech_score"] if tech else edu_cfg["default_non_tech_score"]

    # Research degree bonus (MS/PhD in tech)
    is_research_degree = any(k in degree for k in edu_cfg.get("research_keywords", []))
    if is_research_degree and tech:
        score += edu_cfg.get("research_degree_bonus", 0.10)

    # Research in career descriptions
    career = candidate.get("career_history", [])
    research_kw = edu_cfg.get("research_desc_kw", [])
    if research_kw:
        desc_text = " ".join(r.get("description", "") for r in career).lower()
        if any(k in desc_text for k in research_kw):
            score += 0.05

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Behavioral — simplified, only validated signals
# ---------------------------------------------------------------------------
def get_behavioral_mod(candidate, cfg):
    sig = candidate.get("redrob_signals", {})
    beh = cfg["behavioral"]
    mod = beh.get("base", 0.0)

    for s in beh["signals"]:
        stype = s["type"]

        if stype == "boolean":
            val = sig.get(s["field"])
            mod += s["true_mod"] if val else s["false_mod"]

        elif stype == "days_since":
            raw = sig.get(s["field"], "")
            if raw:
                try:
                    days = (datetime.now() - datetime.strptime(raw, "%Y-%m-%d")).days
                    for band in s["bands"]:
                        lo = band.get("min_days", -1)
                        hi = band.get("max_days", float("inf"))
                        if lo <= days <= hi:
                            mod += band["mod"]
                            break
                except (ValueError, TypeError):
                    pass

        elif stype == "bounded":
            default = s.get("default", 0)
            val = sig.get(s["field"], default)
            if val is None:
                val = default
            if "bands" in s:
                for band in s["bands"]:
                    lo = band.get("min_value", -1)
                    hi = band.get("max_value", float("inf"))
                    if lo <= val <= hi:
                        mod += band["mod"]
                        break

        elif stype == "all_true":
            if all(sig.get(f) for f in s["fields"]):
                mod += s["mod"]

        elif stype == "match":
            val = sig.get(s["field"], "")
            mod += s["mapping"].get(val, 0.0)

    return max(beh["clamp_min"], min(beh["clamp_max"], 1.0 + mod))


# ---------------------------------------------------------------------------
# Honeypots
# ---------------------------------------------------------------------------
def detect_honeypots(candidate, cfg):
    rules_triggered = set()
    rules = cfg["honeypots"]["rules"]
    skills = candidate.get("skills", [])
    ch = candidate.get("career_history", [])
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    edu = candidate.get("education", [])

    for rule in rules:
        if not rule.get("enabled", True):
            continue
        rid = rule["id"]
        rtype = rule["type"]

        if rtype == "advanced_expert_no_duration":
            for s in skills:
                if s.get("proficiency") in ("advanced", "expert") and s.get("duration_months", -1) == 0:
                    rules_triggered.add(rid)
                    break

        elif rtype == "yoe_mismatch":
            total_career_y = sum(r.get("duration_months", 0) for r in ch) / 12
            tol = rule.get("tolerance_years", 3)
            if total_career_y > p.get("years_of_experience", 0) + tol:
                rules_triggered.add(rid)

        elif rtype == "long_role_startup":
            max_y = rule.get("max_years", 12)
            startup_kw = rule.get("startup_keywords", [])
            for r in ch:
                start = r.get("start_date")
                if not start:
                    continue
                try:
                    end = (datetime.now() if r.get("is_current")
                           else datetime.strptime(r.get("end_date", ""), "%Y-%m-%d")
                           if r.get("end_date") else None)
                    if end is None:
                        continue
                    years = (end - datetime.strptime(start, "%Y-%m-%d")).days / 365.25
                    if years > max_y:
                        co = r.get("company", "").lower()
                        if any(k in co for k in startup_kw):
                            rules_triggered.add(rid)
                except (ValueError, TypeError):
                    pass

        elif rtype == "date_overlap":
            non_current = [r for r in ch if not r.get("is_current") and r.get("start_date") and r.get("end_date")]
            for i in range(len(non_current)):
                for j in range(i + 1, len(non_current)):
                    try:
                        s1 = datetime.strptime(non_current[i]["start_date"], "%Y-%m-%d")
                        e1 = datetime.strptime(non_current[i]["end_date"], "%Y-%m-%d")
                        s2 = datetime.strptime(non_current[j]["start_date"], "%Y-%m-%d")
                        e2 = datetime.strptime(non_current[j]["end_date"], "%Y-%m-%d")
                        if s1 < e2 and s2 < e1:
                            rules_triggered.add(rid)
                    except (ValueError, TypeError):
                        pass

        elif rtype == "expert_zero_endorsements":
            count = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0)
            if count >= rule.get("min_count", 5):
                rules_triggered.add(rid)

        elif rtype == "multiple_current_roles":
            max_cur = rule.get("max_current", 1)
            if len([r for r in ch if r.get("is_current", False)]) > max_cur:
                rules_triggered.add(rid)

        elif rtype == "education_yoe_mismatch":
            if edu:
                latest_end = max(e.get("end_year", 0) for e in edu)
                tol = rule.get("tolerance_years", 2)
                if latest_end >= datetime.now().year - 1 and latest_end + p.get("years_of_experience", 0) > datetime.now().year + tol:
                    rules_triggered.add(rid)

        elif rtype == "perfect_score_low_completeness":
            max_comp = rule.get("max_completeness", 30)
            for score in sig.get("skill_assessment_scores", {}).values():
                if score == 100 and sig.get("profile_completeness_score", 100) < max_comp:
                    rules_triggered.add(rid)
                    break

    return rules_triggered


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------
def gen_reasoning(candidate, score_a, base_score, tier_label, skill_scores,
                  score_c, score_d, score_e, score_f, behavioral_mod, cfg):
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    title = p.get("current_title", "N/A")
    yoe = p.get("years_of_experience", 0)
    country = p.get("country", "")
    cid = candidate.get("candidate_id", "")

    import hashlib
    seed = int(hashlib.md5(cid.encode()).hexdigest()[:8], 16)
    rng = __import__("random").Random(seed)

    intro_templates = [
        f"{title} ({yoe}yrs)",
        f"{title} with {yoe} years of experience",
        f"{title}, {yoe}yr professional",
    ]
    parts = [rng.choice(intro_templates)]

    skill_names = {s["name"] for s in skills}
    groups_cfg = cfg["skills"]["groups"]
    evidence = []

    group_labels = {
        "embedding": "embeddings",
        "vector_db": "vector databases",
        "prod_ml": "production ML",
        "eval": "ranking evaluation",
    }

    for gid, g in groups_cfg.items():
        g_score = skill_scores.get(gid, 0.0)
        g_skills = set(g["skills"])
        matched = [s for s in skill_names if s in g_skills]
        if matched and g_score >= 0.20:
            label = group_labels.get(gid, gid)
            evidence.append(f"experienced in {label} ({', '.join(matched[:2])})")

    prod_roles = [
        r for r in career
        if r.get("duration_months", 0) > 12
        and not any(s in r.get("company", "") for s in cfg["services_firms"])
    ]
    if prod_roles:
        evidence.append(f"product engineering at {prod_roles[0]['company']}")

    ai_title_kw = cfg["career"]["ai_title_keywords"]
    if len(career) >= 2:
        sorted_cr = sorted(career, key=lambda x: x.get("start_date", ""))
        first_ai = any(ai in sorted_cr[0].get("title", "").lower() for ai in ai_title_kw)
        last_ai = any(ai in sorted_cr[-1].get("title", "").lower() for ai in ai_title_kw)
        if not first_ai and last_ai:
            evidence.append("career transition into AI/ML roles")

    if tier_label:
        evidence.append(f"{tier_label} background")

    max_ev = cfg["reasoning"].get("max_evidence_items", 2)
    if evidence:
        ev_text = "; ".join(evidence[:max_ev])
        strength_templates = [
            f"Strengths: {ev_text}.",
            f"Key signals: {ev_text}.",
            f"Notable: {ev_text}.",
            f"{ev_text}.",
        ]
        parts.append(rng.choice(strength_templates))

    concerns = []
    notice = sig.get("notice_period_days", 60)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if country != "India" and not sig.get("willing_to_relocate"):
        concerns.append("based outside India, not open to relocation")
    active = sig.get("last_active_date", "")
    if active:
        try:
            d = (datetime.now() - datetime.strptime(active, "%Y-%m-%d")).days
            if d > 90:
                concern_templates = [
                    f"inactive for {d} days",
                    f"{d}-day platform inactivity",
                    f"last active {d} days ago",
                ]
                concerns.append(rng.choice(concern_templates))
        except (ValueError, TypeError):
            pass
    if score_c < 0.2:
        concerns.append("limited career progression data")
    if base_score < 0.20:
        concerns.append("non-engineering career background")

    max_conc = cfg["reasoning"].get("max_concerns", 2)
    if concerns:
        conclusion_templates = [
            f"Concern: {'; '.join(concerns[:max_conc])}.",
            f"Watch for: {'; '.join(concerns[:max_conc])}.",
            f"Note: {'; '.join(concerns[:max_conc])}.",
        ]
        parts.append(rng.choice(conclusion_templates))
    else:
        clean_templates = [
            "Clean profile — no concerns.",
            "Well-rounded candidate.",
            "Strong overall fit.",
        ]
        parts.append(rng.choice(clean_templates))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Multi-view TF-IDF semantic similarity
# ---------------------------------------------------------------------------
def compute_tfidf_scores(candidates, cfg, progress=None):
    sem = cfg["semantic"]
    views = sem.get("views", {"full_profile": {"weight": 1.0, "enabled": True}})
    jd_text = sem["jd_text"]
    n = len(candidates)

    def build_view_texts(vname, vcfg):
        if vname == "full_profile":
            return [_build_profile_text(c) for c in candidates]
        elif vname == "skills":
            return [" ".join(s.get("name", "") for s in c.get("skills", [])) for c in candidates]
        elif vname == "title_headline":
            return [f"{c.get('profile', {}).get('current_title', '')} {c.get('profile', {}).get('headline', '')}" for c in candidates]
        elif vname == "char_ngram":
            source = vcfg.get("text_source", "full_profile")
            if source == "skills_and_headline":
                return [
                    " ".join(s.get("name", "") for s in c.get("skills", []))
                    + " " + c.get("profile", {}).get("current_title", "")
                    + " " + c.get("profile", {}).get("headline", "")
                    for c in candidates
                ]
            return [_build_profile_text(c) for c in candidates]
        return None

    active = [(vn, vc) for vn, vc in views.items() if vc.get("enabled", True)]
    blended = np.zeros(n)
    for vi, (vname, vcfg) in enumerate(active):
        weight = vcfg.get("weight", 1.0)
        texts = build_view_texts(vname, vcfg)
        if texts is None:
            continue
        if progress:
            progress(0.1 + (vi / len(active)) * 0.4, desc=f"TF-IDF: {vname}...")

        kw = {}
        if vcfg.get("analyzer") == "char":
            kw["analyzer"] = "char"
            kw["ngram_range"] = (vcfg.get("ngram_min", 3), vcfg.get("ngram_max", 6))
            kw["max_features"] = vcfg.get("max_features", sem["tfidf_max_features"])
            kw["sublinear_tf"] = False
        else:
            kw["ngram_range"] = (sem["tfidf_ngram_min"], sem["tfidf_ngram_max"])
            kw["max_features"] = sem["tfidf_max_features"]
            kw["stop_words"] = sem.get("tfidf_stop_words", "english")
            kw["sublinear_tf"] = sem.get("tfidf_sublinear_tf", True)

        vec = TfidfVectorizer(**kw)
        all_t = [jd_text] + texts
        t0 = datetime.now()
        tfidf = vec.fit_transform(all_t)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
        blended += scores * weight
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  View '{vname}': {elapsed:.2f}s, mean={scores.mean():.4f}, max={scores.max():.4f}")

    return blended


def compute_tfidf_view(candidates, cfg, vname, vcfg, texts=None):
    """Compute TF-IDF scores for a single view. Returns ndarray of scores."""
    sem = cfg["semantic"]
    jd_text = sem["jd_text"]
    if texts is None:
        if vname == "full_profile":
            texts = [_build_profile_text(c) for c in candidates]
        elif vname == "skills":
            texts = [" ".join(s.get("name", "") for s in c.get("skills", [])) for c in candidates]
        elif vname == "title_headline":
            texts = [f"{c.get('profile', {}).get('current_title', '')} {c.get('profile', {}).get('headline', '')}" for c in candidates]
        elif vname == "char_ngram":
            source = vcfg.get("text_source", "full_profile")
            if source == "skills_and_headline":
                texts = [
                    " ".join(s.get("name", "") for s in c.get("skills", []))
                    + " " + c.get("profile", {}).get("current_title", "")
                    + " " + c.get("profile", {}).get("headline", "")
                    for c in candidates
                ]
            else:
                texts = [_build_profile_text(c) for c in candidates]
        else:
            return np.zeros(len(candidates))
    kw = {}
    if vcfg.get("analyzer") == "char":
        kw["analyzer"] = "char"
        kw["ngram_range"] = (vcfg.get("ngram_min", 3), vcfg.get("ngram_max", 6))
        kw["max_features"] = vcfg.get("max_features", sem["tfidf_max_features"])
        kw["sublinear_tf"] = False
    else:
        kw["ngram_range"] = (sem["tfidf_ngram_min"], sem["tfidf_ngram_max"])
        kw["max_features"] = sem["tfidf_max_features"]
        kw["stop_words"] = sem.get("tfidf_stop_words", "english")
        kw["sublinear_tf"] = sem.get("tfidf_sublinear_tf", True)
    vec = TfidfVectorizer(**kw)
    all_t = [jd_text] + texts
    tfidf = vec.fit_transform(all_t)
    return cosine_similarity(tfidf[0:1], tfidf[1:])[0]


def _build_profile_text(c):
    p = c.get("profile", {})
    parts = [p.get("current_title", ""), p.get("headline", ""), p.get("summary", "")]
    parts.append(" ".join(s.get("name", "") for s in c.get("skills", [])))
    for r in c.get("career_history", []):
        parts.append(f"{r.get('title', '')} at {r.get('company', '')}: {r.get('description', '')}")
    for e in c.get("education", []):
        parts.append(f"{e.get('field_of_study', '')} at {e.get('institution', '')}")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Feature dump
# ---------------------------------------------------------------------------
def dump_top_features(scored, cfg, n=20):
    groups = list(cfg["skills"]["groups"].keys())
    group_headers = [f"{g.upper():<8}" for g in groups]
    header = (
        f"{'ID':<16} {'TITLE':<34} {'SCORE':<7} {'A_TTL':<6} "
        f"{''.join(group_headers)} "
        f"{'C_EXP':<6} {'D_LOC':<6} {'E_EDU':<6} {'F_BEH':<6}"
    )
    print(f"\n{header}")
    print("-" * (160 + 8 * len(groups)))
    for e in scored[:n]:
        c = e["candidate"]
        p = c["profile"]
        group_scores = " ".join(f"{e['skill_scores'].get(g, 0):<8.3f}" for g in groups)
        print(
            f"{e['candidate_id']:<16} {p['current_title'][:33]:<34} {e['score']:<7.4f} "
            f"{e['score_a']:<6.3f} {group_scores} "
            f"{e['score_c']:<6.3f} {e['score_d']:<6.3f} {e['score_e']:<6.3f} {e['behavioral_mod']:<6.3f}"
        )


# ---------------------------------------------------------------------------
# Ranking pipeline
# ---------------------------------------------------------------------------
def rank_candidates(candidates, cfg, tfidf_scores, progress=None):
    weights = cfg["weights"]
    arch = cfg["archetype"]
    results = []

    n = len(candidates)
    for i, c in enumerate(candidates):
        if detect_honeypots(c, cfg):
            continue

        score_a, base_score, tier_label = score_archetype(c, cfg)
        skill_scores, score_b = score_skills_evidence(c, cfg)
        score_c = score_career_pattern(c, cfg)
        score_d = score_location(c, cfg)
        score_e = score_education(c, cfg)
        score_f = tfidf_scores[i] if tfidf_scores is not None else 0.0

        raw = (
            weights["archetype_title"] * score_a
            + weights["skills_evidence"] * score_b
            + weights["career_pattern"] * score_c
            + weights["location"] * score_d
            + weights["education"] * score_e
            + weights.get("semantic_similarity", 0.0) * score_f
        )

        bm = get_behavioral_mod(c, cfg)
        final = min(1.0, max(0.0, raw * bm))

        # Soft gate: let evidence override title caps
        if base_score < 0.50:
            tier = _match_tier(c.get("profile", {}).get("current_title", "").lower(), arch)
            tier_cap = tier["cap"] if tier else arch.get("default_score", 0.10)
            # Evidence factor: 0 (no evidence) to 1 (strong evidence) from skills + semantic
            evidence_factor = min(1.0, score_b * 2.0 + score_f * 3.0)
            soft_cap = base_score + (tier_cap - base_score) * evidence_factor * arch.get("soft_cap_factor", 0.5)
            final = min(final, soft_cap)

        if progress and i % 1000 == 0:
            progress(0.5 + (i / n) * 0.4, desc=f"Scoring candidates ({i:,}/{n:,})...")

        results.append({
            "candidate_id": c["candidate_id"],
            "score": final,
            "score_a": score_a,
            "skill_scores": skill_scores,
            "score_b": score_b,
            "score_c": score_c,
            "score_d": score_d,
            "score_e": score_e,
            "score_f": score_f,
            "behavioral_mod": bm,
            "base_score": base_score,
            "tier_label": tier_label,
            "candidate": c,
        })

    results.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))
    top = results[:100]

    out = []
    for i, r in enumerate(top):
        out.append({
            "candidate_id": r["candidate_id"],
            "rank": i + 1,
            "score": round(r["score"], 6),
            "reasoning": gen_reasoning(
                r["candidate"], r["score_a"], r["base_score"], r["tier_label"],
                r["skill_scores"], r["score_c"], r["score_d"], r["score_e"],
                r.get("score_f", 0.0), r["behavioral_mod"], cfg
            ),
        })

    return out, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Redrob Ranker — config-driven production ranker")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent / "config.json")
    cfg = load_config(config_path)
    print(f"Config: {cfg['meta']['role']} v{cfg['meta']['version']}")
    print(f"  Weights: {cfg['weights']}")
    print(f"  TF-IDF views: {[k for k, v in cfg['semantic'].get('views', {}).items() if v.get('enabled')]}")
    print(f"  Behavioral signals: {len(cfg['behavioral']['signals'])} signals, clamp [{cfg['behavioral']['clamp_min']}, {cfg['behavioral']['clamp_max']}]")

    with open(args.candidates) as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Candidates: {len(candidates)}")

    tfidf_scores = None
    if cfg["semantic"].get("enabled", True):
        print("Computing TF-IDF semantic similarity...")
        tfidf_scores = compute_tfidf_scores(candidates, cfg)
    else:
        print("Semantic similarity disabled")

    print("Ranking...")
    results, all_scored = rank_candidates(candidates, cfg, tfidf_scores)

    if args.dump:
        dump_top_features(all_scored, cfg, 20)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    print(f"Written {len(results)} candidates to {args.out}")


if __name__ == "__main__":
    main()
