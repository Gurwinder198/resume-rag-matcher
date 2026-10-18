"""Thin wrapper around a persistent ChromaDB collection using a local
sentence-transformers embedding model (no API key required)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "resumes"


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_MODEL,
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self.persist_dir = Path(persist_dir)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=self._embed_fn
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add chunks to the store. Each chunk must have "id", "text", and
        "metadata" (a flat dict of str/int/float/bool values)."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def query(self, text: str, top_k: int = 10, where: Optional[dict] = None) -> list[dict]:
        """Return up to top_k matches: [{"id","text","metadata","similarity"}],
        most similar first. "similarity" is in (0, 1] (1 = identical)."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_texts=[text],
            n_results=min(top_k, self.count()),
            where=where,
        )
        matches = []
        for id_, doc, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            similarity = 1.0 / (1.0 + distance)
            matches.append({"id": id_, "text": doc, "metadata": meta, "similarity": similarity})
        return matches

    def count(self) -> int:
        return self._collection.count()
