import nbformat

from scripts.build_notebook import build


def test_notebook_builds_and_validates():
    nb = build()
    nbformat.validate(nb)
    assert len(nb["cells"]) >= 10


def test_notebook_has_required_sections():
    nb = build()
    markdown_text = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in ["Ingest", "chunking", "precision", "Latency", "Conclusions"]:
        assert heading.lower() in markdown_text.lower()


def test_precision_cell_bypasses_hard_filters_to_measure_the_ranker():
    nb = build()
    code_text = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "must_have_skills=[]" in code_text
    assert "min_experience_years=None" in code_text
