"""Evaluate document-level Precision@3 and Recall@3 from actual FAISS hits."""

import json
from pathlib import Path

from part3_support_agent.rag import PolicyRetriever

QUERIES = [
    ("How many days can I return footwear?", {"D02"}),
    ("How long does a COD refund take?", {"D07"}),
    ("Can I return an electronics item?", {"D03", "D12"}),
    ("Is reverse pickup available?", {"D10"}),
    ("What is the delivery SLA?", {"D09"}),
]


def deduplicate(hits: list[dict]) -> list[str]:
    """Map chunks to unique parent document IDs in retrieval order."""
    result = []
    for hit in hits:
        if hit["document_id"] not in result:
            result.append(hit["document_id"])
    return result


def main() -> dict:
    """Run five document-level retrieval evaluations and save arithmetic."""
    retriever = PolicyRetriever()
    retriever.load()
    rows = []
    for query, relevant in QUERIES:
        hits = retriever.retrieve(query, top_k=3, threshold=0.0)
        retrieved = deduplicate(hits)
        overlap = sorted(set(retrieved) & relevant)
        precision = len(overlap) / max(len(retrieved), 1)
        recall = len(overlap) / len(relevant)
        rows.append({
            "query": query, "relevant_documents": sorted(relevant),
            "retrieved_documents": retrieved, "hits": hits,
            "precision_at_3": precision, "recall_at_3": recall,
            "arithmetic": f"Precision@3={len(overlap)}/{len(retrieved)}={precision:.3f}; Recall@3={len(overlap)}/{len(relevant)}={recall:.3f}",
        })
    result = {
        "embedding_model": "all-MiniLM-L6-v2", "document_level": True,
        "queries": rows,
        "average_precision_at_3": sum(row["precision_at_3"] for row in rows) / len(rows),
        "average_recall_at_3": sum(row["recall_at_3"] for row in rows) / len(rows),
    }
    path = Path("knowledge_base/retrieval_eval.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
