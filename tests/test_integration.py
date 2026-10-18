import json

from scripts.generate_dataset import main as generate_dataset
from resume_rag import ResumeRAG
from job_matcher import JobMatcher
from vector_store import VectorStore


def test_full_pipeline_matches_every_generated_job_description(tmp_path):
    generate_dataset(["--out-dir", str(tmp_path)])

    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path / "resumes", vector_store=store)
    ingest_summary = rag.ingest()
    assert ingest_summary["resumes_processed"] >= 30
    assert ingest_summary["resumes_failed"] == 0

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    matcher = JobMatcher(vector_store=store)

    for jd in manifest["job_descriptions"]:
        jd_text = (tmp_path / "job_descriptions" / jd["file"]).read_text(encoding="utf-8")
        result = matcher.match(jd_text, top_k=10)

        assert result["job_description"] == jd_text
        assert isinstance(result["top_matches"], list)
        for match in result["top_matches"]:
            assert 0 <= match["match_score"] <= 100
            for key in ("candidate_name", "resume_path", "matched_skills", "relevant_excerpts", "reasoning"):
                assert key in match

        top_roles = {
            next(r["role"] for r in manifest["resumes"] if r["name"] == m["candidate_name"])
            for m in result["top_matches"][:3]
        }
        assert jd["role"] in top_roles
