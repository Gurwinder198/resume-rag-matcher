from vector_store import VectorStore


def test_add_and_query_returns_most_similar_first(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    store.add_chunks([
        {"id": "a1", "text": "Experienced Python machine learning engineer with PyTorch.",
         "metadata": {"resume_path": "a.txt", "name": "Alice"}},
        {"id": "b1", "text": "Frontend developer skilled in React and TypeScript.",
         "metadata": {"resume_path": "b.txt", "name": "Bob"}},
    ])
    assert store.count() == 2

    results = store.query("Looking for a machine learning engineer with PyTorch experience", top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["name"] == "Alice"
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_query_on_empty_store_returns_empty_list(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    assert store.query("anything") == []


def test_query_respects_where_filter(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_db")
    store.add_chunks([
        {"id": "a1", "text": "Python engineer.", "metadata": {"resume_path": "a.txt", "role": "ml"}},
        {"id": "b1", "text": "Python developer.", "metadata": {"resume_path": "b.txt", "role": "backend"}},
    ])
    results = store.query("Python", top_k=5, where={"role": "ml"})
    assert len(results) == 1
    assert results[0]["metadata"]["resume_path"] == "a.txt"
