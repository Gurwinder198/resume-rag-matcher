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
