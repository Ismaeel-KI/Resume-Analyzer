"""
main.py — FastAPI app + core analyze() pipeline function.
"""

import time
from typing import Any

from fastapi import FastAPI, HTTPException

from app.models import AnalyzeRequest, AnalysisResult
from app.services.parser import parse_file
from app.services.extractor import extract_all
from app.services.matcher import compute_match_score
from app.services.ats import run_ats_analysis
from app.services import llm_gemini


app = FastAPI(
    title="AI Resume ↔ JD Matcher",
    description="Match resumes to job descriptions using Gemini + NLP + embeddings",
    version="1.0.0",
)


def analyze(resume_path: str, jd_path: str) -> dict[str, Any]:
    """
    Core analysis pipeline.

    Steps:
      1. Parse files
      2. NLP extraction (spaCy)
      3. Matching algorithm (sentence-transformers)
      4. ATS optimization
      5. Gemini LLM tasks (structured JSON)
      6. Assemble final result
    """
    start = time.time()

    # ── Step 1: Parse ──────────────────────────────────────────────────────
    resume_text = parse_file(resume_path)
    jd_text = parse_file(jd_path)

    # ── Step 2: NLP Extraction ──────────────────────────────────────────────
    resume_data = extract_all(resume_text, label="resume")
    jd_data = extract_all(jd_text, label="jd")

    # ── Step 3: Matching ────────────────────────────────────────────────────
    match_result = compute_match_score(resume_data, jd_data, resume_text, jd_text)

    # ── Step 4: ATS ─────────────────────────────────────────────────────────
    ats_result = run_ats_analysis(jd_text, resume_text)

    # ── Step 5: Gemini LLM ─────────────────────────────────────────────────
    gemini_resume = llm_gemini.analyze_resume_with_gemini(resume_text, jd_text)
    if "error" in gemini_resume:
        raise ValueError(f"Failed to analyze resume: {gemini_resume.get('error')}")
    
    gemini_jd = llm_gemini.analyze_jd_with_gemini(jd_text)
    if "error" in gemini_jd:
        raise ValueError(f"Failed to analyze JD: {gemini_jd.get('error')}")

    # Merge Gemini skills with NLP skills for richer missing detection
    gemini_required = gemini_jd.get("required_skills", [])
    gemini_resume_skills = gemini_resume.get("skills", [])
    all_missing = list(set(match_result["skills_match"]["missing"]) | set(
        s for s in gemini_required
        if s.lower() not in [x.lower() for x in gemini_resume_skills]
    ))

    improvements = llm_gemini.get_improvement_suggestions(
        resume_text=resume_text,
        jd_text=jd_text,
        match_score=match_result["match_score"],
        missing_skills=all_missing,
    )
    if "error" in improvements:
        improvements = {"strengths": [], "weaknesses": [], "suggestions": []}

    rewrite = llm_gemini.rewrite_resume(
        resume_text=resume_text,
        jd_text=jd_text,
        missing_keywords=ats_result["missing_keywords"],
    )
    if "error" in rewrite:
        rewrite = {"rewritten_resume": "", "changes_made": []}

    # ── Step 6: Assemble ───────────────────────────────────────────────────
    elapsed = round(time.time() - start, 2)

    return {
        "match_score": match_result["match_score"],
        "sub_scores": match_result["sub_scores"],
        "skills_match": match_result["skills_match"],
        "keywords": match_result["keywords"],
        "strengths": improvements.get("strengths", []),
        "weaknesses": improvements.get("weaknesses", []),
        "suggestions": improvements.get("suggestions", []),
        "ats_optimization": ats_result["suggestions"],
        "rewritten_resume": rewrite.get("rewritten_resume", ""),
        "gemini_resume_analysis": gemini_resume,
        "gemini_jd_analysis": gemini_jd,
        "metadata": {
            "resume_path": resume_path,
            "jd_path": jd_path,
            "elapsed_seconds": elapsed,
            "resume_word_count": resume_data["word_count"],
            "jd_word_count": jd_data["word_count"],
            "ats_missing_keywords": ats_result["missing_keywords"],
            "ats_weak_keywords": ats_result["weak_keywords"],
            "interview_red_flags": improvements.get("interview_red_flags", []),
            "rewrite_changes": rewrite.get("changes_made", []),
        },
    }


# ── FastAPI Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AI Resume ↔ JD Matcher API", "version": "1.0.0"}


@app.post("/analyze", response_model=None)
def analyze_endpoint(request: AnalyzeRequest):
    """
    Analyze a resume against a job description.

    Body:
      - resume_path: local path to resume (.pdf / .docx / .txt)
      - jd_path: local path to job description (.pdf / .docx / .txt)
    """
    try:
        result = analyze(request.resume_path, request.jd_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}