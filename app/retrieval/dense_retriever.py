"""
Dense retriever using Qdrant vector search.
Performs ANN (Approximate Nearest Neighbor) search
over 1024-dim BGE-M3 dense embeddings.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    """Single retrieved chunk with its score and metadata."""
    id: str
    text: str
    score: float
    metadata: dict


class DenseRetriever:
    """
    Retrieves top-k chunks using cosine similarity
    over dense BGE-M3 embeddings stored in Qdrant.
    """

    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection = settings.qdrant_collection

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Args:
            query_vector: Dense embedding of the query (1024-dim).
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        top_k = top_k or settings.top_k_retrieval

        results: list[ScoredPoint] = self.client.search(
            collection_name=self.collection,
            query_vector=("dense", query_vector),
            limit=top_k,
            with_payload=True,
        )

        chunks = [
            RetrievedChunk(
                id=str(r.id),
                text=r.payload.get("text", ""),
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k != "text"},
            )
            for r in results
        ]

        logger.info(
            "retrieval.dense.completed",
            top_k=top_k,
            results=len(chunks),
        )

        return chunks