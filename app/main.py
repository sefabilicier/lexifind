"""
LexiFind — Legal RAG Platform
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.observability.logger import setup_logging, get_logger

from app.api.routes import documents
from app.api.routes import documents, query




logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logging()
    logger.info(
        "app.startup",
        model=settings.groq_primary_model,
        embedding=settings.embedding_model,
        qdrant=f"{settings.qdrant_host}:{settings.qdrant_port}",
    )
    yield
    logger.info("LexiFind shutting down", event="app.shutdown")


app = FastAPI(
    title="LexiFind — Legal RAG Platform",
    description="Production-grade RAG system for legal document retrieval and analysis.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "service": "lexi-find",
        "version": "0.1.0",
    }