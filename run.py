"""
run.py — CLI entry point for the AI Resume ↔ JD Matcher.

Usage:
  python run.py --resume data/resume.pdf --jd data/jd.txt
  python run.py --resume data/resume.pdf --jd data/jd.txt --output result.json
  python run.py --resume data/resume.pdf --jd data/jd.txt --no-rewrite
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from app.main import analyze


def main():
    parser = argparse.ArgumentParser(
        description="AI Resume ↔ Job Description Matcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --resume data/resume.pdf --jd data/jd.txt
  python run.py --resume data/resume.docx --jd data/jd.txt --output result.json
  GEMINI_API_KEY=xyz python run.py --resume data/resume.pdf --jd data/jd.txt
        """,
    )
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf, .docx, .txt)")
    parser.add_argument("--jd", required=True, help="Path to job description file (.pdf, .docx, .txt)")
    parser.add_argument("--output", default=None, help="Optional: save JSON result to this file path")
    parser.add_argument("--pretty", action="store_true", default=True, help="Pretty-print JSON output")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the high-level summary (score + skills + suggestions)",
    )

    args = parser.parse_args()

    print(f"\n🔍 Analyzing resume: {args.resume}")
    print(f"📄 Against JD:       {args.jd}")
    print("⏳ Running analysis (this may take 20–60s due to LLM calls)...\n")

    try:
        result = analyze(args.resume, args.jd)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as e:
        print(f"❌ Config error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.summary_only:
        summary = {
            "match_score": result["match_score"],
            "sub_scores": result["sub_scores"],
            "skills_match": result["skills_match"],
            "strengths": result["strengths"],
            "weaknesses": result["weaknesses"],
            "suggestions": result["suggestions"],
            "ats_optimization": result["ats_optimization"],
        }
        output = summary
    else:
        output = result

    indent = 2 if args.pretty else None
    json_str = json.dumps(output, indent=indent, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"✅ Result saved to: {args.output}")
    else:
        print(json_str)

    # Print a quick summary banner
    score = result["match_score"]
    emoji = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
    elapsed = result.get("metadata", {}).get("elapsed_seconds", "?")
    print(f"\n{emoji} Match Score: {score}/100  |  ⏱ {elapsed}s", file=sys.stderr)


if __name__ == "__main__":
    main()