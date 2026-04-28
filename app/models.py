"""
models.py — Pydantic v2 models for request/response validation.
"""

from typing import Any
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    resume_path: str = Field(..., description="Local path to resume file (.pdf, .docx, .txt)")
    jd_path: str = Field(..., description="Local path to job description file (.pdf, .docx, .txt)")


class SkillsMatch(BaseModel):
    matched: list[str] = []
    missing: list[str] = []
    partial: list[str] = []


class KeywordsResult(BaseModel):
    present: list[str] = []
    missing: list[str] = []


class SubScores(BaseModel):
    skills: float
    experience: float
    keywords: float
    education: float


class AnalysisResult(BaseModel):
    match_score: float = Field(..., ge=0, le=100, description="Overall match score 0-100")
    sub_scores: SubScores
    skills_match: SkillsMatch
    keywords: KeywordsResult
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    ats_optimization: list[str] = []
    rewritten_resume: str = ""
    gemini_resume_analysis: dict[str, Any] = {}
    gemini_jd_analysis: dict[str, Any] = {}
    metadata: dict[str, Any] = {}