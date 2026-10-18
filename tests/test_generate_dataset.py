import json

from scripts.generate_dataset import main


def test_generates_30_plus_resumes_across_8_plus_roles_and_5_plus_jds(tmp_path):
    exit_code = main(["--out-dir", str(tmp_path)])
    assert exit_code == 0

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["resumes"]) >= 30
    assert len({r["role"] for r in manifest["resumes"]}) >= 8
    assert len(manifest["job_descriptions"]) >= 5

    resumes_dir = tmp_path / "resumes"
    jd_dir = tmp_path / "job_descriptions"
    assert len(list(resumes_dir.iterdir())) == len(manifest["resumes"])
    assert len(list(jd_dir.iterdir())) == len(manifest["job_descriptions"])


def test_generated_resumes_are_readable_and_non_empty(tmp_path):
    main(["--out-dir", str(tmp_path)])
    import fs_tools

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["resumes"][:5]:
        path = tmp_path / "resumes" / entry["file"]
        result = fs_tools.read_file(str(path))
        assert result["success"] is True
        assert len(result["content"]) > 0


def test_generated_job_descriptions_mention_their_required_skills(tmp_path):
    main(["--out-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["job_descriptions"]:
        text = (tmp_path / "job_descriptions" / entry["file"]).read_text(encoding="utf-8")
        for skill in entry["required_skills"]:
            assert skill in text
