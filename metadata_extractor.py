"""Heuristic extraction of resume metadata: name, skills, experience years,
education. Designed to degrade gracefully — missing sections never raise,
they just produce empty/None values.
"""

from __future__ import annotations

import re
from datetime import datetime

KNOWN_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "SQL", "NoSQL", "React", "Next.js", "Vue", "Angular", "Node.js",
    "Django", "Flask", "FastAPI", "Spring", "PyTorch", "TensorFlow",
    "scikit-learn", "Machine Learning", "Deep Learning", "NLP",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform",
    "Airflow", "Spark", "Kafka", "Snowflake", "dbt", "Tableau",
    "Power BI", "Excel", "Jira", "Git", "CI/CD", "GraphQL", "REST",
    "Linux", "Bash", "Figma", "Jest", "Selenium", "Pandas", "NumPy",
]
_SKILL_LOOKUP = {s.lower(): s for s in KNOWN_SKILLS}

# Single words that are also ordinary English words ("go to the store", "excel
# at", "rest well", "spark ideas", "each spring") — matched case-sensitively
# against their canonical skill casing instead of case-insensitively, since a
# false match here feeds straight into job_matcher's must-have hard filter.
AMBIGUOUS_SKILLS = {"go", "rest", "excel", "spring", "spark"}


def _skill_pattern(lower: str, canonical: str) -> re.Pattern:
    if lower in AMBIGUOUS_SKILLS:
        return re.compile(r"(?<![\w+#.])" + re.escape(canonical) + r"(?![\w+#])")
    return re.compile(r"(?<![\w+#.])" + re.escape(lower) + r"(?![\w+#])", re.IGNORECASE)

_SECTION_RE = {
    "skills": re.compile(
        r"(?:^|\n)\s*(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES)\s*:?\s*\n(.*?)"
        r"(?:\n\s*[A-Z][A-Z ]{2,}\s*:?\s*\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    "education": re.compile(
        r"(?:^|\n)\s*(?:EDUCATION|ACADEMIC BACKGROUND)\s*:?\s*\n(.*?)"
        r"(?:\n\s*[A-Z][A-Z ]{2,}\s*:?\s*\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
}

_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE)
_DATE_RANGE_RE = re.compile(r"(\d{4})\s*[-–—]\s*(\d{4}|present|current)", re.IGNORECASE)


def _looks_like_name(line: str) -> bool:
    words = line.split()
    if not (1 < len(words) <= 4):
        return False
    if any(ch.isdigit() for ch in line):
        return False
    if line.upper() == line:
        return False
    return all(w[0].isupper() for w in words)


def _extract_name(text: str, filename: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            if _looks_like_name(candidate):
                return candidate
            break
    stem = filename.rsplit(".", 1)[0]
    stem = re.sub(r"^resume[_\- ]*", "", stem, flags=re.IGNORECASE)
    cleaned = stem.replace("_", " ").replace("-", " ").strip().title()
    return cleaned or "Unknown"


def _extract_section(text: str, key: str) -> str:
    match = _SECTION_RE[key].search(text)
    return match.group(1).strip() if match else ""


def _extract_skills(text: str) -> list[str]:
    found: dict[str, str] = {}
    skills_section = _extract_section(text, "skills")
    for haystack in (skills_section, text):
        for lower, canonical in _SKILL_LOOKUP.items():
            if canonical.lower() in found:
                continue
            pattern = _skill_pattern(lower, canonical)
            if pattern.search(haystack):
                found[canonical.lower()] = canonical
    return sorted(found.values())


def _extract_experience_years(text: str) -> float | None:
    explicit = [float(m.group(1)) for m in _YEARS_RE.finditer(text)]
    if explicit:
        return max(explicit)

    ranges = _DATE_RANGE_RE.findall(text)
    if not ranges:
        return None
    total_years = 0
    for start, end in ranges:
        end_year = datetime.now().year if end.lower() in {"present", "current"} else int(end)
        total_years += max(0, end_year - int(start))
    return float(total_years) if total_years else None


def _extract_education(text: str) -> list[str]:
    section = _extract_section(text, "education")
    if not section:
        return []
    return [line.strip() for line in section.splitlines() if line.strip()]


def extract_metadata(text: str, filename: str = "") -> dict:
    """Extract {"name", "skills", "experience_years", "education"} from resume text."""
    return {
        "name": _extract_name(text, filename),
        "skills": _extract_skills(text),
        "experience_years": _extract_experience_years(text),
        "education": _extract_education(text),
    }
