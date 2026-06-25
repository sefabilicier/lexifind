"""
Query API endpoint — full pipeline routing support.
POST /api/query
Pipelines: auto | naive | advanced | agentic | corrective
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.config import get_settings
from app.pipeline.router import QueryRouter
from app.pipeline.naive_rag import NaiveRAGPipeline
from app.pipeline.advanced_rag import AdvancedRAGPipeline
from app.pipeline.agentic_rag import AgenticRAGPipeline
from app.pipeline.corrective_rag import CRAGPipeline
from app.observability.logger import get_logger

router = APIRouter(prefix="/api", tags=["Query"])
logger = get_logger(__name__)
settings = get_settings()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    top_n: int = Field(default=3, ge=1, le=10)
    pipeline: str = Field(
        default="auto",
        description="auto | naive | advanced | agentic | corrective"
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    usage: dict
    pipeline: str
    metadata: dict


_PIPELINE_MAP = {
    "naive": NaiveRAGPipeline,
    "advanced": AdvancedRAGPipeline,
    "agentic": AgenticRAGPipeline,
    "corrective": CRAGPipeline,
}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run RAG pipeline for a legal document query.

    Pipeline modes:
    - auto       → QueryRouter classifies and selects best pipeline
    - naive      → retrieve + rerank + generate
    - advanced   → rewrite + multi-retrieve + dedupe + rerank + generate
    - agentic    → LangGraph plan → retrieve → evaluate loop
    - corrective → CRAG grade → self-correct → generate
    """
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Resolve pipeline
        pipeline_name = request.pipeline
        if pipeline_name == "auto":
            query_router = QueryRouter()
            pipeline_name = query_router.route(request.query)

        logger.info(
            "query.received",
            query=request.query[:80],
            pipeline=pipeline_name,
        )

        pipeline_cls = _PIPELINE_MAP.get(pipeline_name)
        if not pipeline_cls:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        pipeline = pipeline_cls(client=client)
        result = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            top_n=request.top_n,
        )

        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            usage=result["usage"],
            pipeline=result["pipeline"],
            metadata={
                k: v for k, v in result.items()
                if k not in {"answer", "citations", "usage", "pipeline"}
            },
        )

    except Exception as e:
        logger.error("query.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))