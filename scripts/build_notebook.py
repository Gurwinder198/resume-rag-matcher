"""Build notebooks/rag_experimentation.ipynb programmatically so its content
stays in sync with the actual pipeline code (resume_rag, job_matcher, metrics).

Run:  python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "rag_experimentation.ipynb"

CODE_CELLS = [
    """import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))

from resume_rag import ResumeRAG
from job_matcher import JobMatcher
from vector_store import VectorStore
from metrics import precision_at_k, measure_latency

BASE = Path.cwd().parent
manifest = json.loads((BASE / "dataset_manifest.json").read_text(encoding="utf-8"))
print(f"{len(manifest['resumes'])} resumes across {len({r['role'] for r in manifest['resumes']})} roles, "
      f"{len(manifest['job_descriptions'])} job descriptions")""",
    """store = VectorStore(persist_dir=BASE / "chroma_db_notebook")
rag = ResumeRAG(resumes_dir=BASE / "resumes", vector_store=store)
ingest_summary = rag.ingest()
ingest_summary""",
    """from chunking import chunk_resume
sample_path = next((BASE / "resumes").glob("*.txt"))
chunks = chunk_resume(sample_path.read_text(encoding="utf-8"))
[(c["section"], len(c["text"])) for c in chunks]""",
    """matcher = JobMatcher(vector_store=store)
role_by_file = {r["file"]: r["role"] for r in manifest["resumes"]}

# must_have_skills=[] / min_experience_years=None bypass the hard must-have
# filter so this measures the semantic+keyword ranker itself, not the filter
# (with the filter on, only the 1-2 candidates meeting every must-have for
# that role can ever appear, which would make precision@5 near-tautological).
precisions = {}
for jd in manifest["job_descriptions"]:
    jd_text = (BASE / "job_descriptions" / jd["file"]).read_text(encoding="utf-8")
    result = matcher.match(jd_text, top_k=10, must_have_skills=[], min_experience_years=None)
    retrieved_paths = [Path(m["resume_path"]).name for m in result["top_matches"]]
    relevant = {f for f, role in role_by_file.items() if role == jd["role"]}
    precisions[jd["file"]] = precision_at_k(retrieved_paths, relevant, k=5)

precisions""",
    """latencies = []
for jd in manifest["job_descriptions"]:
    jd_text = (BASE / "job_descriptions" / jd["file"]).read_text(encoding="utf-8")
    _, elapsed = measure_latency(lambda: matcher.match(jd_text, top_k=10))
    latencies.append(elapsed)

{"mean_latency_s": sum(latencies) / len(latencies), "max_latency_s": max(latencies)}""",
]

MARKDOWN_CELLS = {
    0: "# Resume RAG — Experimentation & Analysis\n\nIngestion, chunking, retrieval accuracy and latency for the resume RAG pipeline (Milestone 2).",
    1: "## 1. Ingest the dataset",
    2: "## 2. Inspect section-aware chunking on a sample resume",
    3: "## 3. Retrieval accuracy — precision@5 per job description\n\nGround truth: a resume is \"relevant\" to a JD if it was generated for that JD's target role (see `dataset_manifest.json`).",
    4: "## 4. Latency — time per job-matching query",
    5: "## 5. Conclusions\n\n- Section-aware chunking keeps Education/Experience/Skills separable for metadata extraction.\n"
       "- Hybrid scoring (semantic + keyword) ranks role-matching candidates above off-role ones.\n"
       "- See the printed precision@5 and latency numbers above for this dataset size.",
}


def build() -> "nbf.NotebookNode":
    nb = nbf.v4.new_notebook()
    cells = []
    for i, code in enumerate(CODE_CELLS):
        if i in MARKDOWN_CELLS:
            cells.append(nbf.v4.new_markdown_cell(MARKDOWN_CELLS[i]))
        cells.append(nbf.v4.new_code_cell(code))
    if len(CODE_CELLS) in MARKDOWN_CELLS:
        cells.append(nbf.v4.new_markdown_cell(MARKDOWN_CELLS[len(CODE_CELLS)]))
    nb["cells"] = cells
    return nb


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbf.write(nb, str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
