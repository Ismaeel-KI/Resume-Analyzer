"""
parser.py — Extract clean text from PDF, DOCX, and TXT files.
"""

import os
from pathlib import Path


def parse_file(file_path: str) -> str:
    """
    Parse a file and return its cleaned text content.
    Supports: .pdf, .docx, .txt
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix == ".txt":
        return _parse_txt(path)
    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Supported: .pdf, .docx, .txt")


def _parse_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing. Run: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text.strip())

    if not text_parts:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return "\n\n".join(text_parts)


def _parse_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing. Run: pip install python-docx")

    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise ValueError(f"No text found in DOCX: {path}")

    return "\n".join(paragraphs)


def _parse_txt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1").strip()

    if not text:
        raise ValueError(f"Empty text file: {path}")

    return text