"""
Graph RAG Pipeline.

Extends Advanced RAG with knowledge graph context:
  1. Standard hybrid retrieval (vector-based)
  2. Entity recognition in query → graph lookup
  3. Neighbor expansion → find related chunk IDs
  4. Community summary → global thematic context
  5. Merge vector + graph contexts → generate

The graph layer catches relationships that pure vector search misses
(e.g. "Article 5 requires X" when the query mentions "X" but not "Article 5").

Reference:
  - Edge et al. (2024) — Microsoft GraphRAG
  - IBM Think: hybrid vector + graph retrieval
"""

from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.embedder import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.dense_retriever import RetrievedChunk
from app.retrieval.reranker import BGEReranker
from app.generation.generator import GroqGenerator
from app.generation.citation_builder import build_citations
from app.graph.builder import LegalKnowledgeGraph
from app.graph.community_detector import CommunityDetector
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class GraphRAGPipeline:
    """
    Graph-enhanced RAG combining vector retrieval with knowledge graph traversal.

    Flow:
        Query
          ↓
        Hybrid Retrieval (vector)
          ↓
        Entity Recognition in Query
          ↓
        Graph Neighbor Expansion → additional chunk IDs
          ↓
        Community Summary → thematic context
          ↓
        Merge + Deduplicate chunks
          ↓
        Rerank → Generate (with graph context injected)
    """

    def __init__(self, client: QdrantClient):
        embedder = BGEEmbedder()
        self.qdrant = client
        self.retriever = HybridRetriever(client=client, embedder=embedder)
        self.reranker = BGEReranker()
        self.generator = GroqGenerator()
        self.graph = LegalKnowledgeGraph()
        self.community_detector = CommunityDetector()

        # Try to load cached graph
        loaded = self.graph.load_cache()
        if not loaded:
            logger.warning(
                "graph_rag.no_cache",
                hint="Call POST /api/graph/build to build the knowledge graph first.",
            )

    def _fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch Qdrant payloads for specific chunk IDs from graph expansion."""
        if not chunk_ids:
            return []

        try:
            results = self.qdrant.retrieve(
                collection_name=settings.qdrant_collection,
                ids=chunk_ids,
                with_payload=True,
            )
            return [
                RetrievedChunk(
                    id=str(r.id),
                    text=r.payload.get("text", ""),
                    score=0.5,  # neutral score for graph-retrieved chunks
                    metadata={
                        k: v for k, v in r.payload.items() if k != "text"
                    },
                )
                for r in results
            ]
        except Exception as e:
            logger.warning("graph_rag.fetch_by_id.failed", error=str(e))
            return []

    def run(self, query: str, top_k: int = 10, top_n: int = 3) -> dict:
        """
        Execute Graph RAG pipeline.

        Args:
            query: User question.
            top_k: Chunks from vector retrieval.
            top_n: Final chunks after reranking.

        Returns:
            Dict with answer, citations, graph metadata, usage.
        """
        logger.info("pipeline.graph_rag.started", query=query[:80])

        # 1. Standard vector retrieval
        vector_chunks = self.retriever.retrieve(query, top_k=top_k)

        # 2. Entity recognition in query
        query_entities = self.graph.find_entities_in_text(query)
        logger.info(
            "graph_rag.entities_found",
            count=len(query_entities),
            entities=query_entities[:5],
        )

        # 3. Graph neighbor expansion
        expanded_entities = []
        for eid in query_entities:
            neighbors = self.graph.get_neighbors(eid, depth=2)
            expanded_entities.extend(neighbors)

        expanded_entities = list(set(expanded_entities))

        # 4. Fetch chunks linked to expanded entities
        graph_chunk_ids = self.graph.get_entity_chunks(expanded_entities)
        graph_chunks = self._fetch_chunks_by_ids(graph_chunk_ids)

        # 5. Community summary for query entities
        communities = self.community_detector.detect(self.graph.graph)
        community_context = ""
        if query_entities and communities:
            # Find which community the query entities belong to
            for comm_id, members in communities.items():
                if any(e in members for e in query_entities):
                    community_context = self.community_detector.summarize_community(
                        members, self.graph.graph
                    )
                    break

        # 6. Subgraph summary for prompt enrichment
        subgraph_summary = self.graph.get_subgraph_summary(
            query_entities + expanded_entities[:10]
        )

        # 7. Merge vector + graph chunks, deduplicate
        seen_ids = set()
        merged = []
        for chunk in vector_chunks + graph_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                merged.append(chunk)

        # 8. Rerank merged pool
        reranked = self.reranker.rerank(query, merged, top_n=top_n)

        # 9. Build enriched context — inject graph knowledge
        context = []
        for chunk in reranked:
            context.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        # Prepend graph context to first chunk for LLM awareness
        if subgraph_summary and context:
            context[0]["text"] = (
                f"{subgraph_summary}\n\n---\n\n{context[0]['text']}"
            )

        # 10. Generate
        result = self.generator.generate(query, context)
        citations = build_citations(reranked)

        logger.info(
            "pipeline.graph_rag.completed",
            vector_chunks=len(vector_chunks),
            graph_chunks=len(graph_chunks),
            merged=len(merged),
            total_tokens=result["usage"]["total_tokens"],
        )

        return {
            "answer": result["answer"],
            "citations": citations,
            "usage": result["usage"],
            "pipeline": "graph_rag",
            "graph_metadata": {
                "query_entities": query_entities,
                "expanded_entities": expanded_entities[:10],
                "graph_chunks_added": len(graph_chunks),
                "community_context": community_context,
                "graph_nodes": self.graph.graph.number_of_nodes(),
                "graph_edges": self.graph.graph.number_of_edges(),
            },
            "chunks_used": len(reranked),
        }