"""Local sentence-transformer and FAISS retrieval."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None

KB_PATH = Path("knowledge_base/policies.json")
INDEX_DIR = Path("artifacts/rag_index")
MODEL_NAME = "all-MiniLM-L6-v2"


class PolicyRetriever:
    """Build and query a document-aware policy vector index."""

    def __init__(self, kb_path: Path = KB_PATH, index_dir: Path = INDEX_DIR):
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
        self.documents = json.loads(kb_path.read_text(encoding="utf-8"))
        if not isinstance(self.documents, list) or len(self.documents) < 12:
            raise ValueError("Knowledge base must contain at least 12 documents")
        self.index_dir = index_dir
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.chunks = [
            {"document_id": doc["document_id"], "document_text": doc["document_text"], "chunk_id": f"{doc['document_id']}_C01"}
            for doc in self.documents
        ]
        self.index = None

    def build(self) -> None:
        """Create a reproducible normalized inner-product FAISS index."""
        if faiss is None:
            raise RuntimeError("FAISS is required to build the policy index; install faiss-cpu in the active interpreter")
        embeddings = self.encoder.encode(
            [chunk["document_text"] for chunk in self.chunks], normalize_embeddings=True
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "policies.faiss"))
        (self.index_dir / "chunks.json").write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load an existing local index and parent-aware chunk metadata."""
        if faiss is None:
            raise RuntimeError("FAISS is required to load the policy index; install faiss-cpu in the active interpreter")
        index_path = self.index_dir / "policies.faiss"
        chunks_path = self.index_dir / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("RAG index is missing; run build() first")
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.35) -> list[dict]:
        """Return scored chunks, retaining document and chunk identity."""
        if self.index is None:
            raise RuntimeError("Retriever index is not loaded")
        vector = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, positions = self.index.search(vector, top_k)
        return [
            {**self.chunks[int(position)], "score": float(score)}
            for score, position in zip(scores[0], positions[0])
            if position >= 0 and float(score) >= threshold
        ]


def build_index() -> None:
    """Rebuild the local policy index."""
    retriever = PolicyRetriever()
    retriever.build()


if __name__ == "__main__":
    build_index()
