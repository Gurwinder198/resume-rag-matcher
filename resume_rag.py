"""Part A — Document Processing Pipeline.

Loads resumes with fs_tools (Milestone 1), chunks them section-aware,
extracts metadata, embeds, and stores everything in a persistent vector DB.

Usage:
    python resume_rag.py --resumes-dir resumes --persist-dir chroma_db
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import fs_tools
from chunking import chunk_resume
from metadata_extractor import extract_metadata
from vector_store import VectorStore

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

# ChromaDB metadata values can't be None, so a resume with no detectable
# years of experience is stored as this sentinel rather than 0.0 — 0.0 would
# be indistinguishable from an actual new grad and would wrongly fail every
# min-years filter. job_matcher treats this sentinel as "unknown, don't filter".
UNKNOWN_EXPERIENCE_YEARS = -1.0


class ResumeRAG:
    def __init__(
        self,
        resumes_dir: str | Path = "resumes",
        persist_dir: str | Path = "chroma_db",
        vector_store: Optional[VectorStore] = None,
    ):
        self.resumes_dir = Path(resumes_dir)
        self.vector_store = vector_store or VectorStore(persist_dir=persist_dir)

    def ingest_file(self, filepath: str) -> dict:
        """Process a single resume file: read, chunk, extract metadata, embed, store.

        Returns {"success": bool, "resume_path": str, "chunks_created": int, "error"?: str}.
        """
        read = fs_tools.read_file(filepath)
        if not read["success"]:
            return {"success": False, "resume_path": filepath, "chunks_created": 0, "error": read["error"]}

        text = read["content"]
        chunks = chunk_resume(text)
        if not chunks:
            warning = read["metadata"].get("warning")
            error = warning or "No content extracted from resume (file may be empty or a scanned image)."
            return {"success": False, "resume_path": filepath, "chunks_created": 0, "error": error}

        metadata = extract_metadata(text, read["metadata"]["name"])

        store_chunks = []
        for chunk in chunks:
            store_chunks.append({
                "id": f"{read['metadata']['name']}::{chunk['chunk_index']}",
                "text": chunk["text"],
                "metadata": {
                    "resume_path": str(read["metadata"]["path"]),
                    "resume_name": read["metadata"]["name"],
                    "section": chunk["section"],
                    "candidate_name": metadata["name"],
                    "skills": ", ".join(metadata["skills"]),
                    "experience_years": (
                        metadata["experience_years"]
                        if metadata["experience_years"] is not None
                        else UNKNOWN_EXPERIENCE_YEARS
                    ),
                    "education": " | ".join(metadata["education"]),
                },
            })
        self.vector_store.add_chunks(store_chunks)
        return {"success": True, "resume_path": filepath, "chunks_created": len(store_chunks)}

    def ingest(self) -> dict:
        """Ingest every supported resume file in resumes_dir.

        Returns {"resumes_processed": int, "resumes_failed": int, "chunks_created": int}.
        """
        processed = failed = chunks_created = 0
        for ext in SUPPORTED_EXTENSIONS:
            for file_info in fs_tools.list_files(str(self.resumes_dir), extension=ext):
                result = self.ingest_file(file_info["path"])
                if result["success"]:
                    processed += 1
                    chunks_created += result["chunks_created"]
                else:
                    failed += 1
        return {"resumes_processed": processed, "resumes_failed": failed, "chunks_created": chunks_created}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest resumes into the vector store.")
    parser.add_argument("--resumes-dir", default="resumes")
    parser.add_argument("--persist-dir", default="chroma_db")
    args = parser.parse_args(argv)

    rag = ResumeRAG(resumes_dir=args.resumes_dir, persist_dir=args.persist_dir)
    summary = rag.ingest()
    print(f"Processed {summary['resumes_processed']} resume(s), "
          f"{summary['chunks_created']} chunk(s), {summary['resumes_failed']} failure(s).")
    return 0 if summary["resumes_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
