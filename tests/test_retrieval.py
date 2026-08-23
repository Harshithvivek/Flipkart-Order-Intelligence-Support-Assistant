"""FAISS retrieval and evaluation artifact tests."""

import json
from pathlib import Path


def test_retrieval_index_and_evaluation():
    assert Path("artifacts/rag_index/policies.faiss").exists()
    result = json.loads(Path("knowledge_base/retrieval_eval.json").read_text())
    assert len(result["queries"]) >= 5
    assert result["average_recall_at_3"] == 1.0
    assert all("arithmetic" in row for row in result["queries"])
