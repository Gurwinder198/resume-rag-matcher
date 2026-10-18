"""Section-aware chunking for resume text.

Resumes are split on recognizable section headers (SUMMARY, SKILLS,
EXPERIENCE, EDUCATION, PROJECTS, CERTIFICATIONS, ...) so a chunk never
mixes, say, Education with Experience. Oversized sections are further
split on paragraph boundaries so no chunk exceeds ``max_chunk_chars``.
"""

from __future__ import annotations

import re

SECTION_HEADERS = [
    "SUMMARY", "OBJECTIVE", "PROFILE",
    "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES",
    "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT",
    "EDUCATION", "ACADEMIC BACKGROUND",
    "PROJECTS", "CERTIFICATIONS", "ACHIEVEMENTS", "PUBLICATIONS",
]

_HEADER_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?\s*$",
    re.IGNORECASE,
)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_name, section_text) pairs, in order."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_name = "GENERAL"
    current_lines: list[str] = []

    for line in lines:
        match = _HEADER_RE.match(line)
        if match:
            sections.append((current_name, current_lines))
            current_name = match.group(1).upper()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_name, current_lines))

    return [
        (name, "\n".join(body_lines).strip())
        for name, body_lines in sections
        if "\n".join(body_lines).strip()
    ]


def _split_oversized(section_text: str, max_chunk_chars: int) -> list[str]:
    """Split long section text into pieces no bigger than max_chunk_chars,
    breaking on blank lines / paragraph boundaries where possible."""
    if len(section_text) <= max_chunk_chars:
        return [section_text]

    paragraphs = re.split(r"\n\s*\n", section_text)
    pieces: list[str] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) > max_chunk_chars and buffer:
            pieces.append(buffer)
            buffer = para
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)

    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chunk_chars:
            final.append(piece)
        else:
            for i in range(0, len(piece), max_chunk_chars):
                final.append(piece[i : i + max_chunk_chars])
    return final


def chunk_resume(text: str, max_chunk_chars: int = 1500) -> list[dict]:
    """Split resume text into section-aware chunks.

    Returns a list of dicts: {"section": str, "text": str, "chunk_index": int}
    ordered as they appear in the resume. "GENERAL" is used for any text
    above the first recognized header, and for resumes with no headers at all.
    """
    if not text or not text.strip():
        return []

    chunks: list[dict] = []
    index = 0
    for name, body in _split_sections(text):
        for piece in _split_oversized(body, max_chunk_chars):
            chunks.append({"section": name, "text": piece, "chunk_index": index})
            index += 1
    return chunks
