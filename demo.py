"""Guided walkthrough for the 3-4 minute demo recording.

Run:  python demo.py   (or demo.bat on Windows)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from job_matcher import JobMatcher
from resume_rag import ResumeRAG
from vector_store import VectorStore

BASE = Path(__file__).resolve().parent
PERSIST_DIR = BASE / "chroma_db"


def _pause(message: str) -> None:
    input(f"\n--- {message} --- (press Enter to continue)")


def scene_ingest(persist_dir: Path = PERSIST_DIR, resumes_dir: Optional[Path] = None) -> VectorStore:
    _pause("Scene 1: Ingest the resume dataset (chunking + embeddings + metadata)")
    store = VectorStore(persist_dir=persist_dir)
    rag = ResumeRAG(resumes_dir=resumes_dir or (BASE / "resumes"), vector_store=store)
    summary = rag.ingest()
    print(f"Ingested {summary['resumes_processed']} resumes into {summary['chunks_created']} chunks "
          f"({summary['resumes_failed']} failures).")
    return store


def scene_match(store: VectorStore, jd_file: str, job_descriptions_dir: Optional[Path] = None) -> None:
    _pause(f"Scene: Match job description {jd_file}")
    jd_dir = job_descriptions_dir or (BASE / "job_descriptions")
    job_description = (jd_dir / jd_file).read_text(encoding="utf-8")
    matcher = JobMatcher(vector_store=store)
    result = matcher.match(job_description, top_k=5)
    print(json.dumps(result, indent=2)[:2000])


def main() -> None:
    store = scene_ingest()
    for jd_file in ("jd_ml_engineer.txt", "jd_frontend_developer.txt", "jd_devops_engineer.txt"):
        scene_match(store, jd_file)
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
