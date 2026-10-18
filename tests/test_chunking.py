from chunking import chunk_resume


def test_headers_split_into_separate_sections():
    text = (
        "Jane Doe\njane@example.com\n\n"
        "SKILLS\nPython, SQL, Docker\n\n"
        "EXPERIENCE\nML Engineer at Acme (2020-2023)\n\n"
        "EDUCATION\nB.Tech CS, IIT Delhi\n"
    )
    chunks = chunk_resume(text)
    sections = [c["section"] for c in chunks]
    assert sections == ["GENERAL", "SKILLS", "EXPERIENCE", "EDUCATION"]
    assert "Python, SQL, Docker" in chunks[1]["text"]


def test_no_headers_falls_back_to_general_section():
    text = "Just a plain paragraph of resume text with no section headers at all."
    chunks = chunk_resume(text)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "GENERAL"


def test_oversized_section_is_split_into_multiple_chunks():
    body = "Worked on projects.\n\n" * 200
    text = "EXPERIENCE\n" + body
    chunks = chunk_resume(text, max_chunk_chars=500)
    experience_chunks = [c for c in chunks if c["section"] == "EXPERIENCE"]
    assert len(experience_chunks) > 1
    assert all(len(c["text"]) <= 500 for c in experience_chunks)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices)


def test_empty_text_returns_no_chunks():
    assert chunk_resume("") == []
    assert chunk_resume("   \n  ") == []
