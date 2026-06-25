"""
Advanced RAG Pipeline.

Extends Naive RAG with:
  1. Query Rewriting — improves retrieval recall
  2. Multi-query Retrieval — fetches from both original + rewritten
  3. Deduplication — merges results, removes duplicate chunk IDs
  4. Reranking — cross-encoder precision pass
  5. Generation — grounded LLM response

Reference:
  - Ma et al. (2023) — "Query Rewriting for Retrieval-Augmented LLMs"
  - AWS RAG best practices: multi-query expansion
  - Google DeepMind: Advanced RAG patterns
"""

from qdrant_client import QdrantClient

from app.ingestion.embedder import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.dense_retriever import RetrievedChunk
from app.retrieval.reranker import BGEReranker
from app.generation.generator import GroqGenerator
from app.generation.citation_builder import build_citations
from app.observability.logger import get_logger

logger = get_logger(__name__)


class AdvancedRAGPipeline:
    """
    Advanced RAG with query rewriting + multi-query retrieval.

    Flow:
        Query
          ↓
        Query Rewrite (fast LLM)
          ↓
        Retrieve(original) + Retrieve(rewritten)
          ↓
        Deduplicate + Merge
          ↓
        Rerank (cross-encoder)
          ↓
        Generate (primary LLM)
    """

    def __init__(self, client: QdrantClient):
        self.embedder = BGEEmbedder()
        self.retriever = HybridRetriever(client=client, embedder=self.embedder)
        self.reranker = BGEReranker()
        self.generator = GroqGenerator()

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove duplicate chunks by ID, keeping highest score."""
        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.id not in seen or chunk.score > seen[chunk.id].score:
                seen[chunk.id] = chunk
        return list(seen.values())

    def run(self, query: str, top_k: int = 10, top_n: int = 3) -> dict:
        """
        Execute advanced RAG pipeline.

        Args:
            query: User question.
            top_k: Chunks to retrieve per query variant.
            top_n: Final chunks to pass to LLM.

        Returns:
            Dict with answer, citations, usage, pipeline info.
        """
        logger.info("pipeline.advanced.started", query=query[:80])

        # 1. Rewrite query for better retrieval
        rewritten_query = self.generator.rewrite_query(query)

        # 2. Retrieve with both original and rewritten queries
        original_chunks = self.retriever.retrieve(query, top_k=top_k)
        rewritten_chunks = self.retriever.retrieve(rewritten_query, top_k=top_k)

        # 3. Merge + deduplicate
        all_chunks = self._deduplicate(original_chunks + rewritten_chunks)

        logger.info(
            "pipeline.advanced.merged",
            original=len(original_chunks),
            rewritten=len(rewritten_chunks),
            deduplicated=len(all_chunks),
        )

        # 4. Rerank merged pool
        reranked = self.reranker.rerank(query, all_chunks, top_n=top_n)

        # 5. Generate
        context = [{"text": c.text, "metadata": c.metadata} for c in reranked]
        result = self.generator.generate(query, context)

        # 6. Citations
        citations = build_citations(reranked)

        logger.info(
            "pipeline.advanced.completed",
            total_tokens=result["usage"]["total_tokens"],
            rewritten_query=rewritten_query[:60],
        )

        return {
            "answer": result["answer"],
            "citations": citations,
            "usage": result["usage"],
            "pipeline": "advanced_rag",
            "original_query": query,
            "rewritten_query": rewritten_query,
            "chunks_retrieved": len(all_chunks),
            "chunks_used": len(reranked),
        }