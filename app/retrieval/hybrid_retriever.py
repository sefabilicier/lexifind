"""
Hybrid retriever using Reciprocal Rank Fusion (RRF).

Combines dense (semantic) + sparse (lexical) results into a single
ranked list without needing score normalization.

Reference:
    Cormack et al. (2009) — "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods"
    IBM Think / AWS RAG best practices: hybrid search as production baseline.
"""

from collections import defaultdict

from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.embedder import BGEEmbedder
from app.retrieval.dense_retriever import DenseRetriever, RetrievedChunk
from app.retrieval.sparse_retriever import SparseRetriever
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# RRF constant — 60 is standard from the original paper
_RRF_K = 60


def _reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    alpha: float,
) -> list[RetrievedChunk]:
    """
    Merge two ranked lists using RRF scoring.

    RRF score = alpha * (1 / (k + dense_rank))
              + (1 - alpha) * (1 / (k + sparse_rank))

    Args:
        dense_results: Results from dense retriever.
        sparse_results: Results from sparse retriever.
        alpha: Weight for dense results (0=pure sparse, 1=pure dense).
               Loaded from config HYBRID_ALPHA (default 0.5).

    Returns:
        Merged and re-ranked list of RetrievedChunk.
    """
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.id] += alpha * (1.0 / (_RRF_K + rank + 1))
        chunks[chunk.id] = chunk

    for rank, chunk in enumerate(sparse_results):
        scores[chunk.id] += (1 - alpha) * (1.0 / (_RRF_K + rank + 1))
        if chunk.id not in chunks:
            chunks[chunk.id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(
            id=chunk_id,
            text=chunks[chunk_id].text,
            score=rrf_score,
            metadata={**chunks[chunk_id].metadata, "rrf_score": rrf_score},
        )
        for chunk_id, rrf_score in ranked
    ]


class HybridRetriever:
    """
    Production-grade hybrid retriever.
    Dense + Sparse → RRF fusion → top-k results.
    """

    def __init__(self, client: QdrantClient, embedder: BGEEmbedder | None = None):
        self.embedder = embedder or BGEEmbedder()
        self.dense = DenseRetriever(client)
        self.sparse = SparseRetriever(client)
        self.alpha = settings.hybrid_alpha

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline for a query string.

        Args:
            query: Raw user query text.
            top_k: Final number of chunks to return after fusion.

        Returns:
            RRF-ranked list of RetrievedChunk.
        """
        top_k = top_k or settings.top_k_retrieval

        logger.info("retrieval.hybrid.started", query=query[:80])

        # 1. Embed query — dense + sparse in one pass
        dense_vec, sparse_weights = self.embedder.embed_query(query)

        # 2. Retrieve from both indexes (fetch more, then fuse)
        fetch_k = top_k * 2
        dense_results = self.dense.retrieve(dense_vec.tolist(), top_k=fetch_k)
        sparse_results = self.sparse.retrieve(sparse_weights, top_k=fetch_k)

        # 3. RRF fusion
        fused = _reciprocal_rank_fusion(dense_results, sparse_results, self.alpha)
        final = fused[:top_k]

        logger.info(
            "retrieval.hybrid.completed",
            dense_hits=len(dense_results),
            sparse_hits=len(sparse_results),
            fused=len(fused),
            returned=len(final),
        )

        return final