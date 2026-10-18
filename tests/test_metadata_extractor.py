from metadata_extractor import extract_metadata

RESUME_TEXT = """John Doe
Machine Learning Engineer | john.doe@example.com

SUMMARY
ML engineer with 5 years of Python experience.

SKILLS
Python, PyTorch, scikit-learn, SQL, Docker, Python

EXPERIENCE
Senior ML Engineer, ShopSmart (2022 - present)
ML Engineer, DataNest (2019 - 2022)

EDUCATION
B.Tech Computer Science, IIT Delhi (2019)
"""


def test_extracts_name_from_first_line():
    meta = extract_metadata(RESUME_TEXT, "resume_john_doe.pdf")
    assert meta["name"] == "John Doe"


def test_extracts_and_dedupes_skills_case_insensitively():
    meta = extract_metadata(RESUME_TEXT, "resume_john_doe.pdf")
    assert "Python" in meta["skills"]
    assert meta["skills"].count("Python") == 1
    assert "Docker" in meta["skills"]


def test_extracts_explicit_years_phrase():
    meta = extract_metadata(RESUME_TEXT, "resume_john_doe.pdf")
    assert meta["experience_years"] == 5.0


def test_extracts_education_lines():
    meta = extract_metadata(RESUME_TEXT, "resume_john_doe.pdf")
    assert any("IIT Delhi" in line for line in meta["education"])


def test_missing_sections_degrade_gracefully():
    meta = extract_metadata("Just some unstructured text.", "resume_unknown.txt")
    assert meta["skills"] == []
    assert meta["education"] == []
    assert meta["experience_years"] is None
    assert meta["name"] == "Unknown"


def test_falls_back_to_filename_when_first_line_is_not_a_name():
    meta = extract_metadata("SKILLS\nPython\n", "resume_alex_chen.txt")
    assert meta["name"] == "Alex Chen"
