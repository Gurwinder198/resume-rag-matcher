# Demo Recording Script (3-4 minutes)

1. (0:00-0:30) Show the project layout: resume_rag.py, job_matcher.py, resumes/
   (32 files), job_descriptions/ (5 files), notebooks/, dataset_manifest.json.
2. (0:30-1:15) Run `demo.bat`. Scene 1 ingests the dataset — call out chunking
   (section-aware), embeddings (local sentence-transformers), ChromaDB storage.
3. (1:15-2:30) Walk through 2-3 job-matching scenes: show the JSON output,
   point at match_score, matched_skills, reasoning, and relevant_excerpts.
4. (2:30-3:15) Open notebooks/rag_experimentation.ipynb: show the precision@5
   table and latency numbers.
5. (3:15-4:00) Run `python job_matcher.py --job-description-file
   job_descriptions/jd_ml_engineer.txt --use-llm-reasoning` to show the
   OpenRouter-generated reasoning text, then wrap up.
