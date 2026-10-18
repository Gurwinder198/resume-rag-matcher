"""Part B — Job Matching Engine.

Semantic + keyword hybrid search over the resume vector store, with
scoring, must-have filtering, and match reasoning.

Usage:
    python job_matcher.py --job-description-file job_descriptions/jd_ml_engineer.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from metadata_extractor import AMBIGUOUS_SKILLS, KNOWN_SKILLS
from resume_rag import UNKNOWN_EXPERIENCE_YEARS
from vector_store import VectorStore

_SKILL_LOOKUP = {s.lower(): s for s in KNOWN_SKILLS}


def _skill_pattern(lower: str, canonical: str) -> re.Pattern:
    if lower in AMBIGUOUS_SKILLS:
        return re.compile(r"(?<![\w+#.])" + re.escape(canonical) + r"(?![\w+#])")
    return re.compile(r"(?<![\w+#.])" + re.escape(lower) + r"(?![\w+#])", re.IGNORECASE)

# A minimum-years requirement is conventionally written with a "+" ("5+
# years") or near a requirement cue word ("requires 5 years", "minimum of
# 5 years", "at least 5 years"). An unanchored "N years" match would also
# fire on incidental mentions like "25 years of company history".
_MIN_YEARS_PLUS_RE = re.compile(r"(\d+)\+\s*years?", re.IGNORECASE)
_MIN_YEARS_CONTEXT_RE = re.compile(r"(?:require[sd]?|minimum|at least)\D{0,20}?(\d+)\s*years?", re.IGNORECASE)
_MUST_HAVE_SECTION_RE = re.compile(
    r"(?:must[- ]have|required skills|requirements?)\s*:?\s*\n?(.*?)(?:\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _find_skills(text: str) -> list[str]:
    found: dict[str, str] = {}
    for lower, canonical in _SKILL_LOOKUP.items():
        pattern = _skill_pattern(lower, canonical)
        if pattern.search(text):
            found[lower] = canonical
    return sorted(found.values())


def extract_requirements(job_description: str) -> dict:
    """Pull hard requirements out of a JD: mentioned skills (for scoring),
    explicit must-have skills (for hard filtering), and minimum experience years."""
    section_match = _MUST_HAVE_SECTION_RE.search(job_description)
    years = [float(m.group(1)) for m in _MIN_YEARS_PLUS_RE.finditer(job_description)]
    if not years:
        years = [float(m.group(1)) for m in _MIN_YEARS_CONTEXT_RE.finditer(job_description)]
    return {
        "mentioned_skills": _find_skills(job_description),
        "must_have_skills": _find_skills(section_match.group(1)) if section_match else [],
        "min_experience_years": max(years) if years else None,
    }


def _group_by_candidate(hits: list[dict]) -> dict[str, dict]:
    """Collapse chunk-level vector-store hits into one entry per resume_path,
    keeping the best-matching chunk per section and the top similarity."""
    candidates: dict[str, dict] = {}
    for hit in hits:
        meta = hit["metadata"]
        path = meta["resume_path"]
        entry = candidates.setdefault(path, {
            "resume_path": path,
            "candidate_name": meta["candidate_name"],
            "skills": [s.strip() for s in meta["skills"].split(",") if s.strip()],
            "experience_years": meta["experience_years"],
            "best_similarity": 0.0,
            "sections": {},
        })
        entry["best_similarity"] = max(entry["best_similarity"], hit["similarity"])
        existing = entry["sections"].get(meta["section"])
        if existing is None or hit["similarity"] > existing["similarity"]:
            entry["sections"][meta["section"]] = {"text": hit["text"], "similarity": hit["similarity"]}
    return candidates


def _build_reasoning(candidate: dict, matched_skills: list[str], semantic_pct: float) -> str:
    sections = ", ".join(sorted(candidate["sections"])) or "no section"
    skills_part = f"matching skills ({', '.join(matched_skills)})" if matched_skills else "general relevance"
    return (
        f"{candidate['candidate_name']} matched on {sections}, with {semantic_pct:.0f}% semantic "
        f"similarity to the job description and {skills_part}."
    )


def _build_llm_reasoning(client: Any, model: str, job_description: str, candidate: dict, matched_skills: list[str]) -> str:
    excerpt = next(iter(candidate["sections"].values()))["text"][:500] if candidate["sections"] else ""
    prompt = (
        f"Job description:\n{job_description}\n\n"
        f"Candidate: {candidate['candidate_name']}\n"
        f"Matched skills: {', '.join(matched_skills) or 'none'}\n"
        f"Resume excerpt:\n{excerpt}\n\n"
        "In 1-2 sentences, explain why this candidate is a good match for the role."
    )
    response = client.chat.send(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=150)
    return response.choices[0].message.content.strip()


class JobMatcher:
    def __init__(self, vector_store: VectorStore, llm_client: Any = None, model: Optional[str] = None):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    def match(
        self,
        job_description: str,
        top_k: int = 10,
        must_have_skills: Optional[list[str]] = None,
        min_experience_years: Optional[float] = None,
        use_llm_reasoning: bool = False,
    ) -> dict:
        """Run semantic + hybrid search and return the ranked output JSON."""
        requirements = extract_requirements(job_description)
        required_skills = must_have_skills if must_have_skills is not None else requirements["must_have_skills"]
        min_years = min_experience_years if min_experience_years is not None else requirements["min_experience_years"]
        mentioned_skills = requirements["mentioned_skills"]

        hits = self.vector_store.query(job_description, top_k=max(top_k * 5, 20))
        candidates = _group_by_candidate(hits)

        scored = []
        for candidate in candidates.values():
            if required_skills and not all(
                any(req.lower() == s.lower() for s in candidate["skills"]) for req in required_skills
            ):
                continue
            exp_years = candidate["experience_years"]
            if min_years is not None and exp_years != UNKNOWN_EXPERIENCE_YEARS and exp_years < min_years:
                continue

            matched_skills = sorted(
                {s for s in candidate["skills"] if s.lower() in {m.lower() for m in mentioned_skills}}
            )
            keyword_ratio = len(matched_skills) / len(mentioned_skills) if mentioned_skills else 0.0
            semantic_pct = candidate["best_similarity"] * 100
            match_score = round(min(100.0, semantic_pct * 0.7 + keyword_ratio * 100 * 0.3), 1)

            if use_llm_reasoning and self.llm_client is not None:
                reasoning = _build_llm_reasoning(self.llm_client, self.model, job_description, candidate, matched_skills)
            else:
                reasoning = _build_reasoning(candidate, matched_skills, semantic_pct)

            scored.append({
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": match_score,
                "matched_skills": matched_skills,
                "relevant_excerpts": [s["text"][:300] for s in candidate["sections"].values()],
                "reasoning": reasoning,
            })

        scored.sort(key=lambda c: c["match_score"], reverse=True)
        return {"job_description": job_description, "top_matches": scored[:top_k]}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Match a job description against ingested resumes.")
    parser.add_argument("--job-description", help="Job description text")
    parser.add_argument("--job-description-file", help="Path to a file containing the job description")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--must-have-skills", default=None, help="Comma-separated list, overrides auto-detection")
    parser.add_argument("--min-experience-years", type=float, default=None)
    parser.add_argument("--use-llm-reasoning", action="store_true")
    args = parser.parse_args(argv)

    if args.job_description_file:
        job_description = Path(args.job_description_file).read_text(encoding="utf-8")
    elif args.job_description:
        job_description = args.job_description
    else:
        parser.error("Provide --job-description or --job-description-file")
        return 2

    llm_client = None
    if args.use_llm_reasoning:
        from dotenv import load_dotenv

        load_dotenv()
        from openrouter import OpenRouter

        llm_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    # None means "flag omitted" (auto-detect must-haves from the JD); an
    # explicit "" means "override with no must-haves" and must not fall
    # through to auto-detection just because the string is falsy.
    must_have = (
        [s.strip() for s in args.must_have_skills.split(",") if s.strip()]
        if args.must_have_skills is not None
        else None
    )
    matcher = JobMatcher(vector_store=VectorStore(persist_dir=args.persist_dir), llm_client=llm_client)
    result = matcher.match(
        job_description,
        top_k=args.top_k,
        must_have_skills=must_have,
        min_experience_years=args.min_experience_years,
        use_llm_reasoning=args.use_llm_reasoning,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
