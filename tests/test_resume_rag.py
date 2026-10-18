from resume_rag import ResumeRAG
from vector_store import VectorStore

RESUME_A = """Alice Nguyen
Machine Learning Engineer

SUMMARY
ML engineer with 6 years of Python experience.

SKILLS
Python, PyTorch, Machine Learning

EXPERIENCE
Senior ML Engineer, Acme (2019 - present)

EDUCATION
MSc Computer Science, Stanford (2018)
"""

RESUME_B = """Bob Carter
Frontend Developer

SKILLS
JavaScript, React, TypeScript

EXPERIENCE
Frontend Developer, Webify (2021 - present)

EDUCATION
BSc Information Technology, UCLA (2020)
"""


def test_ingest_processes_all_resumes_and_populates_vector_store(tmp_path):
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "resume_alice.txt").write_text(RESUME_A, encoding="utf-8")
    (resumes_dir / "resume_bob.txt").write_text(RESUME_B, encoding="utf-8")

    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=resumes_dir, vector_store=store)
    summary = rag.ingest()

    assert summary["resumes_processed"] == 2
    assert summary["resumes_failed"] == 0
    assert summary["chunks_created"] > 0
    assert store.count() == summary["chunks_created"]


def test_ingest_file_extracts_candidate_metadata(tmp_path):
    resume_path = tmp_path / "resume_alice.txt"
    resume_path.write_text(RESUME_A, encoding="utf-8")

    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path, vector_store=store)
    result = rag.ingest_file(str(resume_path))

    assert result["success"] is True
    assert result["chunks_created"] > 0

    hit = store.query("Python machine learning", top_k=1)[0]
    assert hit["metadata"]["candidate_name"] == "Alice Nguyen"
    assert "Python" in hit["metadata"]["skills"]
    assert hit["metadata"]["experience_years"] == 6.0


def test_ingest_on_empty_directory_returns_zero_counts(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path / "empty", vector_store=store)
    summary = rag.ingest()
    assert summary == {"resumes_processed": 0, "resumes_failed": 0, "chunks_created": 0}


def test_ingest_file_with_no_extractable_text_is_reported_as_failed(tmp_path):
    empty_resume = tmp_path / "resume_empty.txt"
    empty_resume.write_text("   \n  ", encoding="utf-8")

    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path, vector_store=store)
    result = rag.ingest_file(str(empty_resume))

    assert result["success"] is False
    assert "error" in result


def test_ingest_counts_empty_resume_as_failed_not_processed(tmp_path):
    (tmp_path / "resume_alice.txt").write_text(RESUME_A, encoding="utf-8")
    (tmp_path / "resume_empty.txt").write_text("   \n  ", encoding="utf-8")

    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path, vector_store=store)
    summary = rag.ingest()

    assert summary["resumes_processed"] == 1
    assert summary["resumes_failed"] == 1
