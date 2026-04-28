"""
extractor.py — NLP extraction of skills, experience, keywords
using spaCy for initial pass, structured for Gemini refinement.
"""

import re
from typing import Any

# Lazy-load spaCy model
_nlp = None

SKILL_PATTERNS = [
    # Programming languages
    r"\b(python|java|javascript|typescript|c\+\+|c#|go|rust|kotlin|swift|scala|r|matlab)\b",
    # Web / frameworks
    r"\b(react|angular|vue|node\.?js|django|flask|fastapi|spring|express|nextjs|nuxtjs)\b",
    # Cloud / DevOps
    r"\b(aws|gcp|azure|docker|kubernetes|terraform|ansible|jenkins|gitlab|github actions)\b",
    # Databases
    r"\b(postgresql|mysql|mongodb|redis|elasticsearch|cassandra|dynamodb|sqlite|oracle)\b",
    # ML / Data
    r"\b(tensorflow|pytorch|scikit-learn|pandas|numpy|spark|hadoop|kafka|airflow|mlflow)\b",
    # Concepts
    r"\b(machine learning|deep learning|nlp|computer vision|data science|devops|ci/cd|rest api|graphql|microservices|agile|scrum)\b",
]

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s+years?\s+(?:of\s+)?(?:experience|exp\.?)",
    r"(?:experience|exp\.?)\s+(?:of\s+)?(\d+)\+?\s+years?",
    r"(\d+)\+?\s+years?\s+(?:in|with|using)",
]

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e", "mba",
    "b.sc", "m.sc", "degree", "university", "college", "institute", "graduation",
    "computer science", "information technology", "engineering", "data science",
]


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Auto-download if missing
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            raise ImportError("spaCy is required. Run: pip install spacy")
    return _nlp


def extract_skills(text: str) -> list[str]:
    """Extract technical skills using regex patterns."""
    text_lower = text.lower()
    found = set()
    for pattern in SKILL_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found.update(m.strip().lower() for m in matches if m.strip())
    return sorted(found)


def extract_experience_years(text: str) -> dict[str, Any]:
    """Extract years of experience and role titles."""
    years_found = []
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        years_found.extend(int(m) for m in matches if m.isdigit())

    total_years = max(years_found) if years_found else 0

    # Extract job titles via spaCy NER
    nlp = get_nlp()
    doc = nlp(text[:5000])  # limit for performance
    roles = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "WORK_OF_ART"):
            # Skip — we want roles, not names
            pass

    # Simple heuristic: lines with job-title words
    title_keywords = r"\b(engineer|developer|scientist|analyst|manager|lead|architect|designer|consultant|director|intern|associate|senior|junior|principal|staff)\b"
    title_lines = []
    for line in text.split("\n"):
        if re.search(title_keywords, line, re.IGNORECASE) and len(line.strip()) < 100:
            title_lines.append(line.strip())

    return {
        "total_years": total_years,
        "roles": list(set(title_lines))[:10],
    }


def extract_keywords(text: str) -> list[str]:
    """Extract important noun phrases and technical keywords via spaCy."""
    nlp = get_nlp()
    doc = nlp(text[:8000])

    keywords = set()

    # Noun chunks (multi-word technical terms)
    for chunk in doc.noun_chunks:
        token = chunk.text.strip().lower()
        if 2 <= len(token) <= 50 and not chunk.root.is_stop:
            keywords.add(token)

    # Named entities
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "GPE", "LANGUAGE", "LAW"):
            keywords.add(ent.text.strip().lower())

    # Single important tokens
    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.pos_ in ("NOUN", "PROPN")
            and len(token.text) > 2
        ):
            keywords.add(token.lemma_.lower())

    # Filter out generic noise
    noise = {"resume", "cv", "job", "description", "candidate", "company", "work", "year", "experience"}
    keywords -= noise

    return sorted(keywords)[:50]


def extract_education(text: str) -> list[str]:
    """Extract education-related lines."""
    lines = text.split("\n")
    edu_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in EDUCATION_KEYWORDS):
            clean = line.strip()
            if clean and len(clean) > 5:
                edu_lines.append(clean)
    return edu_lines[:10]


def extract_all(text: str, label: str = "document") -> dict[str, Any]:
    """Run full NLP extraction pipeline on a text."""
    return {
        "label": label,
        "skills": extract_skills(text),
        "experience": extract_experience_years(text),
        "keywords": extract_keywords(text),
        "education": extract_education(text),
        "char_count": len(text),
        "word_count": len(text.split()),
    }