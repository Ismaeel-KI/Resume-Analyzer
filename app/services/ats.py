"""
ats.py — ATS keyword optimization.
Extracts high-value JD keywords and flags what's missing/weak in the resume.
"""

import re
from collections import Counter
from typing import Any


# Boost weight for these high-signal terms
ATS_PRIORITY_TERMS = {
    "required", "must have", "must-have", "mandatory", "essential",
    "key skills", "primary", "core", "critical", "strong",
}

# Common filler words to exclude
STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "are", "have", "will",
    "you", "our", "your", "their", "from", "into", "able", "good", "work",
    "team", "role", "join", "like", "also", "make", "help", "use", "new",
    "we", "be", "an", "a", "in", "to", "of", "is", "at", "on",
    "candidate", "job", "position", "company", "experience", "skills",
    "looking", "seeking", "excellent", "strong", "proven", "preferred",
}


def extract_ats_keywords(jd_text: str) -> list[dict[str, Any]]:
    """
    Extract ATS-relevant keywords from JD with priority scores.
    Returns list of {keyword, priority, context}.
    """
    lines = jd_text.split("\n")
    keyword_scores: dict[str, float] = {}
    keyword_context: dict[str, str] = {}

    for line in lines:
        line_lower = line.lower()
        priority_boost = 1.5 if any(t in line_lower for t in ATS_PRIORITY_TERMS) else 1.0

        # Extract 1-3 word phrases (ngrams)
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b", line)
        words_lower = [w.lower() for w in words]

        # Unigrams
        for w in words_lower:
            if w not in STOP_TERMS and len(w) > 2:
                keyword_scores[w] = keyword_scores.get(w, 0) + (1.0 * priority_boost)
                keyword_context.setdefault(w, line.strip()[:120])

        # Bigrams
        for i in range(len(words_lower) - 1):
            bg = f"{words_lower[i]} {words_lower[i+1]}"
            if words_lower[i] not in STOP_TERMS and words_lower[i+1] not in STOP_TERMS:
                keyword_scores[bg] = keyword_scores.get(bg, 0) + (1.8 * priority_boost)
                keyword_context.setdefault(bg, line.strip()[:120])

    # Sort by score, take top 40
    top = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:40]

    return [
        {
            "keyword": kw,
            "score": round(score, 2),
            "context": keyword_context.get(kw, ""),
        }
        for kw, score in top
    ]


def identify_weak_keywords(
    ats_keywords: list[dict],
    resume_text: str,
) -> dict[str, list[str]]:
    """
    Compare ATS keywords against resume text.
    Returns: missing (not in resume), weak (mentioned once/briefly).
    """
    resume_lower = resume_text.lower()

    missing_kw = []
    weak_kw = []

    for item in ats_keywords:
        kw = item["keyword"]
        count = resume_lower.count(kw)

        if count == 0:
            missing_kw.append(kw)
        elif count == 1:
            weak_kw.append(kw)
        # count >= 2 → well-represented, skip

    return {
        "missing_keywords": missing_kw[:20],
        "weak_keywords": weak_kw[:20],
    }


def build_ats_suggestions(
    ats_keywords: list[dict],
    resume_text: str,
    jd_text: str,
) -> list[str]:
    """
    Generate concrete ATS optimization suggestions.
    """
    analysis = identify_weak_keywords(ats_keywords, resume_text)
    suggestions = []

    missing = analysis["missing_keywords"]
    weak = analysis["weak_keywords"]

    if missing:
        top_missing = missing[:5]
        suggestions.append(
            f"Add these high-priority keywords missing from your resume: {', '.join(top_missing)}"
        )

    if weak:
        top_weak = weak[:5]
        suggestions.append(
            f"Strengthen usage of these keywords (mentioned only once): {', '.join(top_weak)}"
        )

    # Check for action verbs
    action_verbs = ["developed", "built", "designed", "led", "managed", "improved", "implemented", "architected"]
    resume_lower = resume_text.lower()
    missing_verbs = [v for v in action_verbs if v not in resume_lower]
    if missing_verbs:
        suggestions.append(
            f"Use stronger action verbs in bullet points: {', '.join(missing_verbs[:4])}"
        )

    # Check quantification
    if not re.search(r"\d+%|\d+ times|reduced by|improved by|increased by", resume_lower):
        suggestions.append(
            "Add quantified achievements (e.g., 'Improved performance by 30%', 'Scaled to 1M users')"
        )

    # Check summary section
    if "summary" not in resume_lower and "objective" not in resume_lower and "profile" not in resume_lower:
        suggestions.append(
            "Add a professional summary/profile section at the top with role-specific keywords"
        )

    return suggestions


def run_ats_analysis(jd_text: str, resume_text: str) -> dict[str, Any]:
    """Full ATS analysis pipeline."""
    ats_keywords = extract_ats_keywords(jd_text)
    kw_analysis = identify_weak_keywords(ats_keywords, resume_text)
    suggestions = build_ats_suggestions(ats_keywords, resume_text, jd_text)

    return {
        "top_jd_keywords": [item["keyword"] for item in ats_keywords[:20]],
        "missing_keywords": kw_analysis["missing_keywords"],
        "weak_keywords": kw_analysis["weak_keywords"],
        "suggestions": suggestions,
    }