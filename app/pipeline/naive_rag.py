"""
Naive RAG Pipeline.

The simplest RAG pattern:
    Query → Retrieve (hybrid) → Rerank → Generate

No query transformation, no validation.
Used as baseline and for simple factual lookups.

Reference: Lewis et al. (2020) — "Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks"
"""

from qdrant_client import QdrantClient

from app.ingestion.embedder import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import BGEReranker
from app.generation.generator import GroqGenerator
from app.generation.citation_builder import build_citations
from app.observability.logger import get_logger

logger = get_logger(__name__)


class NaiveRAGPipeline:
    """
    Baseline RAG: retrieve → rerank → generate.
    No query rewriting, no self-correction.
    """

    def __init__(self, client: QdrantClient):
        self.embedder = BGEEmbedder()
        self.retriever = HybridRetriever(client=client, embedder=self.embedder)
        self.reranker = BGEReranker()
        self.generator = GroqGenerator()

    def run(self, query: str, top_k: int = 10, top_n: int = 3) -> dict:
        """
        Execute naive RAG pipeline.

        Args:
            query: User question.
            top_k: Chunks to retrieve.
            top_n: Chunks to pass to LLM after reranking.

        Returns:
            Dict with answer, citations, usage, pipeline info.
        """
        logger.info("pipeline.naive.started", query=query[:80])

        # 1. Retrieve
        chunks = self.retriever.retrieve(query, top_k=top_k)

        # 2. Rerank
        reranked = self.reranker.rerank(query, chunks, top_n=top_n)

        # 3. Generate
        context = [{"text": c.text, "metadata": c.metadata} for c in reranked]
        result = self.generator.generate(query, context)

        # 4. Build citations
        citations = build_citations(reranked)

        logger.info(
            "pipeline.naive.completed",
            total_tokens=result["usage"]["total_tokens"],
        )

        return {
            "answer": result["answer"],
            "citations": citations,
            "usage": result["usage"],
            "pipeline": "naive_rag",
            "chunks_retrieved": len(chunks),
            "chunks_used": len(reranked),
        }