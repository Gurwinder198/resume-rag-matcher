import fs_tools


def test_read_file_returns_content_and_metadata(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Jane Doe\nSKILLS\nPython, SQL\n", encoding="utf-8")

    result = fs_tools.read_file(str(path))

    assert result["success"] is True
    assert "Jane Doe" in result["content"]
    assert result["metadata"]["name"] == "resume.txt"
    assert result["metadata"]["word_count"] > 0


def test_read_file_missing_file_returns_error():
    result = fs_tools.read_file("does_not_exist.txt")
    assert result["success"] is False
    assert "error" in result


def test_list_files_filters_by_extension(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")

    files = fs_tools.list_files(str(tmp_path), extension=".txt")

    assert len(files) == 1
    assert files[0]["name"] == "a.txt"


def test_list_files_on_missing_directory_returns_empty_list():
    assert fs_tools.list_files("no/such/dir") == []


def test_write_file_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "out.txt"
    result = fs_tools.write_file(str(target), "hello")
    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == "hello"


def test_search_in_file_finds_case_insensitive_matches(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Experienced in Python and PYTHON tooling.", encoding="utf-8")

    result = fs_tools.search_in_file(str(path), "python")

    assert result["success"] is True
    assert result["match_count"] == 2
