"""
Corrective RAG (CRAG) Pipeline.

CRAG adds a self-correction loop to standard RAG:
  1. Retrieve chunks
  2. Grade each chunk — relevant | irrelevant | ambiguous
  3. If ALL chunks are irrelevant → rewrite query and retry once
  4. If SOME chunks are relevant → filter and generate
  5. If confident → generate directly

Reference:
  Yan et al. (2024) — "Corrective Retrieval Augmented Generation"
  https://arxiv.org/abs/2401.15884
"""

from groq import Groq
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.embedder import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.dense_retriever import RetrievedChunk
from app.retrieval.reranker import BGEReranker
from app.generation.generator import GroqGenerator
from app.generation.citation_builder import build_citations
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_GRADER_PROMPT = """You are a relevance grader for a legal RAG system.

Given a QUESTION and a DOCUMENT CHUNK, assess relevance.

Respond with ONLY one word:
- relevant   : chunk directly answers or strongly supports the question
- ambiguous  : chunk is partially related but not sufficient alone
- irrelevant : chunk has no meaningful relation to the question
"""


class CRAGPipeline:
    """
    Corrective RAG with chunk-level relevance grading and query rewriting fallback.

    Flow:
        Retrieve
          ↓
        Grade each chunk (fast LLM)
          ↓
        All irrelevant? → Rewrite query → Retrieve again
          ↓
        Filter relevant + ambiguous chunks
          ↓
        Rerank → Generate
    """

    def __init__(self, client: QdrantClient):
        embedder = BGEEmbedder()
        self.retriever = HybridRetriever(client=client, embedder=embedder)
        self.reranker = BGEReranker()
        self.generator = GroqGenerator()
        self.grader_client = Groq(api_key=settings.groq_api_key)

    def _grade_chunk(self, query: str, chunk_text: str) -> str:
        """
        Grade a single chunk for relevance to the query.
        Returns: 'relevant' | 'ambiguous' | 'irrelevant'
        """
        response = self.grader_client.chat.completions.create(
            model=settings.groq_fast_model,
            messages=[
                {"role": "system", "content": _GRADER_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION: {query}\n\nDOCUMENT CHUNK:\n{chunk_text[:800]}",
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().lower()
        valid = {"relevant", "ambiguous", "irrelevant"}
        return label if label in valid else "ambiguous"

    def _grade_all(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Grade all retrieved chunks.
        Returns dict grouped by grade label.
        """
        graded: dict[str, list] = {
            "relevant": [],
            "ambiguous": [],
            "irrelevant": [],
        }
        for chunk in chunks:
            label = self._grade_chunk(query, chunk.text)
            graded[label].append(chunk)
            logger.info(
                "crag.grade",
                label=label,
                chunk_id=chunk.id[:8],
                score=round(chunk.score, 3),
            )
        return graded

    def run(self, query: str, top_k: int = 10, top_n: int = 3) -> dict:
        """
        Execute CRAG pipeline with self-correction.

        Args:
            query: User question.
            top_k: Chunks to retrieve.
            top_n: Final chunks after reranking.

        Returns:
            Dict with answer, citations, grade summary, usage.
        """
        logger.info("pipeline.crag.started", query=query[:80])

        # 1. Initial retrieval
        chunks = self.retriever.retrieve(query, top_k=top_k)

        # 2. Grade chunks
        graded = self._grade_all(query, chunks)

        grade_summary = {k: len(v) for k, v in graded.items()}
        logger.info("crag.grade_summary", **grade_summary)

        # 3. Self-correction: if all irrelevant → rewrite + re-retrieve
        usable = graded["relevant"] + graded["ambiguous"]

        if not usable:
            logger.info("crag.correcting", reason="all_chunks_irrelevant")
            rewritten = self.generator.rewrite_query(query)
            chunks = self.retriever.retrieve(rewritten, top_k=top_k)
            graded = self._grade_all(rewritten, chunks)
            usable = graded["relevant"] + graded["ambiguous"]
            grade_summary["correction_applied"] = True
            grade_summary["rewritten_query"] = rewritten

        # 4. Fallback — use all chunks if still nothing usable
        if not usable:
            logger.warning("crag.fallback", reason="no_usable_after_correction")
            usable = chunks

        # 5. Rerank usable chunks
        reranked = self.reranker.rerank(query, usable, top_n=top_n)

        # 6. Generate
        context = [{"text": c.text, "metadata": c.metadata} for c in reranked]
        result = self.generator.generate(query, context)
        citations = build_citations(reranked)

        logger.info(
            "pipeline.crag.completed",
            total_tokens=result["usage"]["total_tokens"],
            usable_chunks=len(usable),
        )

        return {
            "answer": result["answer"],
            "citations": citations,
            "usage": result["usage"],
            "pipeline": "corrective_rag",
            "grade_summary": grade_summary,
            "chunks_used": len(reranked),
        }