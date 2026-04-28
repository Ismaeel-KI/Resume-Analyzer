"""
matcher.py — Compute match score between resume and JD.

Scoring weights:
  Skills similarity   → 40%
  Experience relevance → 30%
  Keyword overlap     → 20%
  Education           → 10%
"""

from typing import Any
import numpy as np

# Lazy-load embedding model
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("sentence-transformers is required. Run: pip install sentence-transformers")
    return _embed_model


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def embed_text(text: str) -> np.ndarray:
    model = get_embed_model()
    return model.encode(text, convert_to_numpy=True)


def embed_list(items: list[str]) -> np.ndarray:
    """Embed a list of strings into a single averaged vector."""
    if not items:
        return np.zeros(384)  # all-MiniLM-L6-v2 dim
    model = get_embed_model()
    vecs = model.encode(items, convert_to_numpy=True)
    return vecs.mean(axis=0)


# ── Skills matching ──────────────────────────────────────────────────────────

def match_skills(resume_skills: list[str], jd_skills: list[str]) -> dict[str, Any]:
    """
    Classify each JD skill as matched / missing / partial in resume.
    Returns structured result + a 0–100 score.
    """
    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)

    matched, missing, partial = [], [], []

    for skill in jd_set:
        if skill in resume_set:
            matched.append(skill)
        else:
            # Check partial: any resume skill contains this skill or vice versa
            is_partial = any(
                skill in rs or rs in skill
                for rs in resume_set
            )
            if is_partial:
                partial.append(skill)
            else:
                missing.append(skill)

    total_jd = len(jd_set)
    if total_jd == 0:
        score = 50.0  # neutral if JD has no detectable skills
    else:
        score = ((len(matched) + 0.5 * len(partial)) / total_jd) * 100

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "partial": sorted(partial),
        "score": round(min(score, 100), 1),
    }


# ── Experience relevance ──────────────────────────────────────────────────────

def score_experience(
    resume_exp: dict,
    jd_exp: dict,
    resume_text: str,
    jd_text: str,
) -> float:
    """
    Score experience relevance 0–100.
    Combines: years comparison + semantic embedding similarity of role context.
    """
    resume_years = resume_exp.get("total_years", 0)
    jd_years = jd_exp.get("total_years", 0)

    # Years score: resume meets or exceeds JD requirement
    if jd_years == 0:
        years_score = 70.0  # no requirement stated → neutral
    elif resume_years >= jd_years:
        years_score = 100.0
    else:
        years_score = (resume_years / jd_years) * 100

    # Semantic score: compare resume roles context vs JD roles context
    resume_roles_text = " ".join(resume_exp.get("roles", [])) or resume_text[:1000]
    jd_roles_text = " ".join(jd_exp.get("roles", [])) or jd_text[:1000]

    vec_r = embed_text(resume_roles_text)
    vec_j = embed_text(jd_roles_text)
    semantic_score = cosine_similarity(vec_r, vec_j) * 100

    return round(0.5 * years_score + 0.5 * semantic_score, 1)


# ── Keyword overlap ───────────────────────────────────────────────────────────

def score_keywords(resume_keywords: list[str], jd_keywords: list[str]) -> dict[str, Any]:
    """
    Measure keyword overlap + semantic similarity of keyword sets.
    Returns score 0–100 and present/missing keyword lists.
    """
    resume_kw = set(k.lower() for k in resume_keywords)
    jd_kw = set(k.lower() for k in jd_keywords)

    present = sorted(resume_kw & jd_kw)
    missing = sorted(jd_kw - resume_kw)

    if not jd_kw:
        overlap_score = 50.0
    else:
        overlap_score = (len(present) / len(jd_kw)) * 100

    # Semantic boost: embed both keyword sets
    if resume_kw and jd_kw:
        vec_r = embed_list(list(resume_kw))
        vec_j = embed_list(list(jd_kw))
        sem = cosine_similarity(vec_r, vec_j) * 100
        score = round(0.6 * overlap_score + 0.4 * sem, 1)
    else:
        score = round(overlap_score, 1)

    return {
        "present": present[:30],
        "missing": missing[:30],
        "score": min(score, 100),
    }


# ── Education ─────────────────────────────────────────────────────────────────

def score_education(resume_edu: list[str], jd_text: str) -> float:
    """
    Simple education score: check if resume education satisfies JD requirements.
    """
    if not resume_edu:
        return 40.0  # penalise if nothing found

    edu_text = " ".join(resume_edu).lower()
    jd_lower = jd_text.lower()

    # Detect required degree level in JD
    if "phd" in jd_lower or "doctorate" in jd_lower:
        required = "phd"
    elif "master" in jd_lower or "m.tech" in jd_lower or "m.sc" in jd_lower:
        required = "master"
    elif "bachelor" in jd_lower or "b.tech" in jd_lower or "b.e" in jd_lower:
        required = "bachelor"
    else:
        required = None

    if required is None:
        return 80.0  # no specific requirement → good

    degree_hierarchy = ["bachelor", "master", "phd"]
    candidate_level = 0
    for i, level in enumerate(degree_hierarchy):
        synonyms = {
            "bachelor": ["bachelor", "b.tech", "b.e", "b.sc", "undergraduate"],
            "master": ["master", "m.tech", "m.e", "m.sc", "mba", "postgraduate"],
            "phd": ["phd", "doctorate", "doctoral"],
        }[level]
        if any(s in edu_text for s in synonyms):
            candidate_level = i

    required_level = degree_hierarchy.index(required)

    if candidate_level >= required_level:
        return 90.0
    elif candidate_level == required_level - 1:
        return 60.0
    else:
        return 30.0


# ── Final score ───────────────────────────────────────────────────────────────

def compute_match_score(
    resume_data: dict,
    jd_data: dict,
    resume_text: str,
    jd_text: str,
) -> dict[str, Any]:
    """
    Compute the final weighted match score and sub-scores.
    """
    skills_result = match_skills(
        resume_data.get("skills", []),
        jd_data.get("skills", []),
    )

    exp_score = score_experience(
        resume_data.get("experience", {}),
        jd_data.get("experience", {}),
        resume_text,
        jd_text,
    )

    kw_result = score_keywords(
        resume_data.get("keywords", []),
        jd_data.get("keywords", []),
    )

    edu_score = score_education(
        resume_data.get("education", []),
        jd_text,
    )

    # Weighted final score
    final = (
        0.40 * skills_result["score"]
        + 0.30 * exp_score
        + 0.20 * kw_result["score"]
        + 0.10 * edu_score
    )

    return {
        "match_score": round(min(final, 100), 1),
        "sub_scores": {
            "skills": skills_result["score"],
            "experience": exp_score,
            "keywords": kw_result["score"],
            "education": edu_score,
        },
        "skills_match": {
            "matched": skills_result["matched"],
            "missing": skills_result["missing"],
            "partial": skills_result["partial"],
        },
        "keywords": {
            "present": kw_result["present"],
            "missing": kw_result["missing"],
        },
    }