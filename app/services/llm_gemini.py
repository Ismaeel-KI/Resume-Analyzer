"""
llm_gemini.py — Gemini integration with structured JSON prompt templates.

All prompts are designed to return valid JSON.
Constraints:
  - No hallucinated experience
  - Concise outputs
  - Structured schema enforced in prompt
"""

import json
import os
import re
from typing import Any


def _get_model():
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai is required. Run: pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before running: export GEMINI_API_KEY=your_key"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def _call_gemini(prompt: str) -> str:
    """Call Gemini and return raw text response."""
    model = _get_model()
    response = model.generate_content(prompt)
    return response.text


def _parse_json_response(raw: str) -> dict | list:
    """Safely parse JSON from Gemini response, stripping markdown fences."""
    if not raw or not raw.strip():
        return {"error": "Empty response from Gemini"}
    
    # Strip ```json ... ``` or ``` ... ```
    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    
    # Try to find JSON object in response
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract JSON object if wrapped in text
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Return error dict
        return {"error": f"JSON parse failed", "raw": clean[:200]}


# ── Prompt Templates ──────────────────────────────────────────────────────────

RESUME_ANALYSIS_PROMPT = """You are an ATS (Applicant Tracking System) analyzer.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd}

Analyze this resume against the job description. Return ONLY a valid JSON object (no markdown, no explanation).

{{
  "match_score": <number 0-100>,
  "sub_scores": {{"skills": <number>, "experience": <number>, "keywords": <number>, "education": <number>}},
  "skills_match": {{"matched": <list>, "missing": <list>, "partial": <list>}},
  "keywords": {{"present": <list>, "missing": <list>}},
  "strengths": <list of 3-4 bullet points>,
  "weaknesses": <list of 3-4 bullet points>
}}

Rules: Output ONLY valid JSON. No explanation, no markdown. Keep lists short (max 10 items). Be accurate, no hallucinations."""


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_resume_with_gemini(resume_text: str, jd_text: str = "") -> dict[str, Any]:
    """Use Gemini to extract structured data from resume text."""
    prompt = RESUME_ANALYSIS_PROMPT.format(resume_text=resume_text[:6000], jd=jd_text[:6000])
    raw = _call_gemini(prompt)
    return _parse_json_response(raw)


def analyze_jd_with_gemini(jd_text: str) -> dict[str, Any]:
    """Extract required skills and key requirements from job description."""
    prompt = f"""Analyze this job description. Return ONLY valid JSON (no explanation).

JOB DESCRIPTION:
{jd_text[:6000]}

{{
  "required_skills": <list of technical skills>,
  "key_requirements": <list of 5 key requirements>,
  "experience_years": <number or null>,
  "education_required": <string or "not specified">,
  "responsibilities": <list of 3 main responsibilities>
}}

Output ONLY JSON. No markdown, no explanation."""
    raw = _call_gemini(prompt)
    return _parse_json_response(raw)


def get_improvement_suggestions(
    resume_text: str,
    jd_text: str,
    match_score: float,
    missing_skills: list[str],
) -> dict[str, Any]:
    """Get actionable improvement suggestions from Gemini."""
    missing_str = ", ".join(missing_skills[:10]) if missing_skills else "none identified"
    prompt = f"""Based on this resume vs job description, provide improvements. Return ONLY valid JSON.

RESUME:
{resume_text[:5000]}

JOB DESCRIPTION:
{jd_text[:5000]}

Match Score: {match_score}/100
Missing Skills: {missing_str}

{{
  "strengths": <list of 3 strengths>,
  "weaknesses": <list of 3 weaknesses>,
  "suggestions": <list of 5 specific improvements>
}}

Output ONLY JSON. No explanation."""
    raw = _call_gemini(prompt)
    return _parse_json_response(raw)


def rewrite_resume(
    resume_text: str,
    jd_text: str,
    missing_keywords: list[str],
) -> dict[str, Any]:
    """Rewrite resume to better match job description."""
    keywords_str = ", ".join(missing_keywords[:15]) if missing_keywords else "none"
    prompt = f"""Rewrite this resume to match the job description better. Return ONLY valid JSON.

RESUME:
{resume_text[:5000]}

JOB DESCRIPTION:
{jd_text[:5000]}

Keywords to include: {keywords_str}

{{
  "rewritten_resume": <string>,
  "changes_made": <list of 3-4 changes>
}}

Output ONLY JSON. No explanation."""
    raw = _call_gemini(prompt)
    return _parse_json_response(raw)
