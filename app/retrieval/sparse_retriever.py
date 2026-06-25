"""
Sparse retriever using Qdrant sparse vector search.
Uses BGE-M3 lexical weights (SPLADE-style) for exact term matching.
Complements dense search by catching keyword-precise legal terms.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NamedSparseVector

from app.config import get_settings
from app.observability.logger import get_logger
from app.retrieval.dense_retriever import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


class SparseRetriever:
    """
    Retrieves top-k chunks using sparse lexical weights from BGE-M3.
    Critical for legal documents where exact term matching matters
    (e.g. specific law numbers, article references, legal jargon).
    """

    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection = settings.qdrant_collection

    def retrieve(
        self,
        sparse_weights: dict,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Args:
            sparse_weights: Dict of {token_id: weight} from BGE-M3.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        top_k = top_k or settings.top_k_retrieval

        if not sparse_weights:
            logger.warning("retrieval.sparse.empty_weights")
            return []

        indices = [int(k) for k in sparse_weights.keys()]
        values = [float(v) for v in sparse_weights.values()]

        results = self.client.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=indices, values=values),
            ),
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
            "retrieval.sparse.completed",
            top_k=top_k,
            results=len(chunks),
        )

        return chunks