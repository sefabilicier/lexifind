"""
Citation builder — formats source references in the final response.
Ensures every answer is traceable back to source documents.
"""

from app.retrieval.dense_retriever import RetrievedChunk


def build_citations(chunks: list[RetrievedChunk]) -> list[dict]:
    """
    Build a structured citation list from retrieved chunks.

    Returns:
        List of citation dicts with source, chunk_index, score.
    """
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append({
            "ref": i,
            "source": chunk.metadata.get("source", "unknown"),
            "chunk_index": chunk.metadata.get("chunk_index", "?"),
            "reranker_score": chunk.metadata.get("reranker_score", chunk.score),
        })
    return citations