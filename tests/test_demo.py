def test_demo_scenes_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    import demo

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "resume_a.txt").write_text(
        "Alice\n\nSKILLS\nPython\n\nEXPERIENCE\nEngineer (2020 - present)\n", encoding="utf-8"
    )

    store = demo.scene_ingest(persist_dir=tmp_path / "chroma_db", resumes_dir=resumes_dir)
    assert store.count() > 0

    jd_dir = tmp_path / "job_descriptions"
    jd_dir.mkdir()
    (jd_dir / "jd_test.txt").write_text("Looking for a Python engineer.", encoding="utf-8")

    demo.scene_match(store, "jd_test.txt", job_descriptions_dir=jd_dir)
