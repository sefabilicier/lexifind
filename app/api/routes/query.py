"""
Query API endpoint.
POST /api/query — hybrid retrieval + rerank + LLM generation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.embedder import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import BGEReranker
from app.observability.logger import get_logger

router = APIRouter(prefix="/api", tags=["Query"])
logger = get_logger(__name__)
settings = get_settings()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    top_n: int = Field(default=3, ge=1, le=10)
    pipeline: str = Field(default="hybrid", description="hybrid | dense | sparse")


class ChunkResponse(BaseModel):
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    chunks: list[ChunkResponse]
    pipeline_used: str


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Retrieve relevant chunks for a query using hybrid search + reranking.
    """
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        embedder = BGEEmbedder()
        retriever = HybridRetriever(client=client, embedder=embedder)
        reranker = BGEReranker()

        logger.info(
            "query.received",
            query=request.query[:80],
            pipeline=request.pipeline,
        )

        # 1. Hybrid retrieval
        chunks = retriever.retrieve(request.query, top_k=request.top_k)

        # 2. Rerank
        reranked = reranker.rerank(request.query, chunks, top_n=request.top_n)

        logger.info(
            "query.completed",
            returned=len(reranked),
        )

        return QueryResponse(
            query=request.query,
            chunks=[
                ChunkResponse(
                    text=c.text,
                    score=round(c.score, 4),
                    metadata=c.metadata,
                )
                for c in reranked
            ],
            pipeline_used="hybrid+rerank",
        )

    except Exception as e:
        logger.error("query.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))