"""
Query API endpoint — full pipeline routing support.
POST /api/query
Pipelines: auto | naive | advanced | agentic | corrective | graph
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.config import get_settings
from app.pipeline.router import QueryRouter
from app.pipeline.naive_rag import NaiveRAGPipeline
from app.pipeline.advanced_rag import AdvancedRAGPipeline
from app.pipeline.agentic_rag import AgenticRAGPipeline
from app.pipeline.corrective_rag import CRAGPipeline
from app.pipeline.graph_rag import GraphRAGPipeline
from app.security.prompt_guard import PromptGuard
from app.security.content_filter import ContentFilter
from app.api.middleware.rate_limit import limiter
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
        description="auto | naive | advanced | agentic | corrective | graph"
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
    "graph": GraphRAGPipeline,
}


@router.post("/query", response_model=QueryResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query(request: Request, body: QueryRequest):
    """
    Run RAG pipeline for a legal document query.

    Pipeline modes:
    - auto       → QueryRouter classifies and selects best pipeline
    - naive      → retrieve + rerank + generate
    - advanced   → rewrite + multi-retrieve + dedupe + rerank + generate
    - agentic    → LangGraph plan → retrieve → evaluate loop
    - corrective → CRAG grade → self-correct → generate
    - graph      → knowledge graph + vector hybrid retrieval
    """
    try:
        # ── 1. Prompt injection guard ──────────────────────────────────────────
        guard = PromptGuard()
        guard_result = guard.check(body.query)

        if not guard_result.is_safe:
            logger.warning(
                "security.injection.blocked",
                reason=guard_result.reason,
                method=guard_result.method,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Query blocked by security filter: {guard_result.reason}",
            )

        # ── 2. Resolve pipeline ────────────────────────────────────────────────
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        pipeline_name = body.pipeline
        if pipeline_name == "auto":
            query_router = QueryRouter()
            pipeline_name = query_router.route(body.query)

        logger.info(
            "query.received",
            query=body.query[:80],
            pipeline=pipeline_name,
        )

        pipeline_cls = _PIPELINE_MAP.get(pipeline_name)
        if not pipeline_cls:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # ── 3. Run pipeline ────────────────────────────────────────────────────
        pipeline_instance = pipeline_cls(client=client)
        result = pipeline_instance.run(
            query=body.query,
            top_k=body.top_k,
            top_n=body.top_n,
        )

        # ── 4. Output content filter ───────────────────────────────────────────
        content_filter = ContentFilter()
        filter_result = content_filter.filter(
            result["answer"],
            result["citations"],
        )

        logger.info(
            "query.completed",
            pipeline=pipeline_name,
            output_safe=filter_result.is_safe,
            warnings=len(filter_result.warnings),
        )

        return QueryResponse(
            answer=filter_result.filtered_answer,
            citations=result["citations"],
            usage=result["usage"],
            pipeline=result["pipeline"],
            metadata={
                **{
                    k: v for k, v in result.items()
                    if k not in {"answer", "citations", "usage", "pipeline"}
                },
                "security": {
                    "output_safe": filter_result.is_safe,
                    "warnings": filter_result.warnings,
                },
            },
        )

    except HTTPException:
        raise  # security ve validation hatalarını olduğu gibi geçir
    except Exception as e:
        logger.error("query.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))