"""
LexiFind — Legal RAG Platform
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.observability.logger import setup_logging, get_logger
from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from app.api.routes import documents, query

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(
        "app.startup",
        model=settings.groq_primary_model,
        embedding=settings.embedding_model,
        qdrant=f"{settings.qdrant_host}:{settings.qdrant_port}",
    )
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="LexiFind — Legal RAG Platform",
    description="Production-grade RAG system for legal document retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Rate limiter state ─────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ── Middleware (order matters — last added = first executed) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key auth — reads valid keys from .env
_valid_keys = {
    k.strip()
    for k in settings.api_keys.split(",")
    if k.strip()
}
app.add_middleware(APIKeyMiddleware, valid_keys=_valid_keys)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/api/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "lexi-find", "version": "0.1.0"}