import json

from job_matcher import JobMatcher, extract_requirements
from vector_store import VectorStore
from resume_rag import ResumeRAG

JD_ML = """We are hiring a Senior ML Engineer.

Must have:
Python, PyTorch, Machine Learning

Requires 5+ years of experience.
"""

RESUME_ML = """Alice Nguyen
Machine Learning Engineer

SUMMARY
ML engineer with 6 years of Python experience building PyTorch models.

SKILLS
Python, PyTorch, Machine Learning

EXPERIENCE
Senior ML Engineer, Acme (2018 - present)

EDUCATION
MSc Computer Science, Stanford (2018)
"""

RESUME_FRONTEND = """Bob Carter
Frontend Developer

SKILLS
JavaScript, React, TypeScript

EXPERIENCE
Frontend Developer, Webify (2021 - present)

EDUCATION
BSc Information Technology, UCLA (2020)
"""


def _build_matcher(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path, vector_store=store)
    (tmp_path / "resume_alice.txt").write_text(RESUME_ML, encoding="utf-8")
    (tmp_path / "resume_bob.txt").write_text(RESUME_FRONTEND, encoding="utf-8")
    rag.ingest()
    return JobMatcher(vector_store=store)


def test_extract_requirements_parses_must_have_and_min_years():
    req = extract_requirements(JD_ML)
    assert "Python" in req["must_have_skills"]
    assert req["min_experience_years"] == 5.0


def test_extract_requirements_handles_jd_with_no_explicit_requirements():
    req = extract_requirements("We need a great teammate who loves to learn.")
    assert req["must_have_skills"] == []
    assert req["min_experience_years"] is None


def test_extract_requirements_does_not_match_ambiguous_words_in_ordinary_prose():
    jd = "We'd like someone ready to go and rest easy while sparking ideas each spring."
    req = extract_requirements(jd)
    assert req["mentioned_skills"] == []


def test_extract_requirements_ignores_incidental_year_counts():
    jd = (
        "We are hiring a Python developer.\n\n"
        "Our company has 25 years of history in the industry."
    )
    req = extract_requirements(jd)
    assert req["min_experience_years"] is None


def test_match_ranks_ml_candidate_above_frontend_candidate(tmp_path):
    matcher = _build_matcher(tmp_path)
    result = matcher.match(JD_ML, top_k=10)

    assert result["job_description"] == JD_ML
    assert len(result["top_matches"]) == 1
    top = result["top_matches"][0]
    assert top["candidate_name"] == "Alice Nguyen"
    assert 0 <= top["match_score"] <= 100
    assert "Python" in top["matched_skills"]
    assert top["reasoning"]


def test_must_have_filter_excludes_candidates_missing_required_skill(tmp_path):
    matcher = _build_matcher(tmp_path)
    result = matcher.match(
        "Looking for a Go developer.", top_k=10, must_have_skills=["Go"], min_experience_years=None,
    )
    assert result["top_matches"] == []


def test_min_experience_filter_excludes_under_qualified_candidates(tmp_path):
    matcher = _build_matcher(tmp_path)
    result = matcher.match(
        "React developer needed.", top_k=10, must_have_skills=[], min_experience_years=100,
    )
    assert result["top_matches"] == []


RESUME_NO_YEARS_STATED = """Charlie Doe
Backend Developer

SKILLS
Java, Spring

EXPERIENCE
Backend Developer, SomeCo

EDUCATION
BSc Computer Science
"""


def test_min_experience_filter_does_not_exclude_candidates_with_unknown_experience(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path, vector_store=store)
    (tmp_path / "resume_charlie.txt").write_text(RESUME_NO_YEARS_STATED, encoding="utf-8")
    rag.ingest()

    matcher = JobMatcher(vector_store=store)
    result = matcher.match("Java developer needed.", top_k=5, must_have_skills=[], min_experience_years=5)

    assert len(result["top_matches"]) == 1
    assert result["top_matches"][0]["candidate_name"] == "Charlie Doe"


def test_llm_reasoning_uses_injected_client(tmp_path):
    class FakeMessage:
        content = "Great fit because of strong PyTorch experience."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeChat:
        def send(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        chat = FakeChat()

    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    rag = ResumeRAG(resumes_dir=tmp_path, vector_store=store)
    (tmp_path / "resume_alice.txt").write_text(RESUME_ML, encoding="utf-8")
    rag.ingest()

    matcher = JobMatcher(vector_store=store, llm_client=FakeClient(), model="openai/gpt-4o-mini")
    result = matcher.match(JD_ML, top_k=5, use_llm_reasoning=True)
    assert result["top_matches"][0]["reasoning"] == "Great fit because of strong PyTorch experience."


def test_cli_empty_must_have_skills_disables_auto_detected_filter(tmp_path, capsys):
    (tmp_path / "resume_alice.txt").write_text(RESUME_ML, encoding="utf-8")
    persist_dir = tmp_path / "chroma_db"
    ResumeRAG(resumes_dir=tmp_path, persist_dir=persist_dir).ingest()

    from job_matcher import main as cli_main
    jd_requiring_go = "We need a Go developer.\n\nMust have:\nGo\n"
    exit_code = cli_main([
        "--job-description", jd_requiring_go,
        "--persist-dir", str(persist_dir),
        "--must-have-skills", "",
    ])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    # Alice has no "Go" skill; an explicit empty override must disable the
    # auto-detected "Go" must-have, not fall through to auto-detection.
    assert len(payload["top_matches"]) == 1


def test_cli_prints_valid_json_output(tmp_path, capsys):
    (tmp_path / "resume_alice.txt").write_text(RESUME_ML, encoding="utf-8")
    persist_dir = tmp_path / "chroma_db"
    ResumeRAG(resumes_dir=tmp_path, persist_dir=persist_dir).ingest()

    from job_matcher import main as cli_main
    exit_code = cli_main(["--job-description", JD_ML, "--persist-dir", str(persist_dir), "--top-k", "5"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["job_description"] == JD_ML
    assert isinstance(payload["top_matches"], list)
    if payload["top_matches"]:
        match = payload["top_matches"][0]
        for key in ("candidate_name", "resume_path", "match_score", "matched_skills", "relevant_excerpts", "reasoning"):
            assert key in match
