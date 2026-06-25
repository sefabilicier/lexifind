"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Reranking is a second-pass scoring step:
  - Retriever: fast, approximate (ANN)
  - Reranker: slow, precise (query × chunk cross-attention)

This significantly improves precision for the final N chunks
passed to the LLM, reducing hallucination risk.

Reference: Cohere / IBM Think RAG pipeline best practices.
"""

from FlagEmbedding import FlagReranker

from app.config import get_settings
from app.observability.logger import get_logger
from app.retrieval.dense_retriever import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


class BGEReranker:
    """
    Singleton reranker using BAAI/bge-reranker-v2-m3.
    Takes top-k retrieved chunks and returns top-n re-scored chunks.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(
            "reranker.loading",
            model="BAAI/bge-reranker-v2-m3",
        )

        self.model = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
        )
        self._initialized = True
        logger.info("reranker.ready")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks using cross-encoder scoring.

        Args:
            query: Original user query.
            chunks: Chunks from hybrid retriever.
            top_n: How many to keep after reranking.

        Returns:
            top_n chunks sorted by reranker score descending.
        """
        top_n = top_n or settings.final_n_rerank

        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores: list[float] = self.model.compute_score(pairs, normalize=True)
        
        logger.info(
            "reranker.scores",
            raw_scores=[round(s, 4) for s in scores],
            top_n=top_n,
        )

        # Attach reranker scores
        for chunk, score in zip(chunks, scores):
            chunk.metadata["reranker_score"] = round(score, 4)
            chunk.score = score

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        final = reranked[:top_n]

        logger.info(
            "reranking.completed",
            input_chunks=len(chunks),
            top_n=top_n,
            top_score=round(final[0].score, 4) if final else 0,
        )

        return final