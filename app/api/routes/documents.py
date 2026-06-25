"""
Document ingestion API endpoint.
POST /api/documents/ingest — upload and ingest a file.
"""

from pathlib import Path
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.observability.logger import get_logger

from app.graph.builder import LegalKnowledgeGraph
from app.retrieval.dense_retriever import RetrievedChunk


router = APIRouter(prefix="/api/documents", tags=["Documents"])
logger = get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm"}


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the vector store.
    Supported formats: PDF, DOCX, HTML
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        pipeline = IngestionPipeline(qdrant_client=client)
        result = pipeline.ingest(tmp_path)
        result["original_filename"] = file.filename
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()  # tam stack trace konsola yazdır
        logger.error("ingestion.failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/graph/build")
async def build_graph():
    """
    Build the knowledge graph from all ingested chunks in Qdrant.
    Run this after ingesting documents.
    """
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

        # Scroll all chunks from Qdrant
        all_chunks = []
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for r in results:
                all_chunks.append({
                    "id": str(r.id),
                    "text": r.payload.get("text", ""),
                })
            if offset is None:
                break

        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks found. Ingest documents first.",
            )

        graph = LegalKnowledgeGraph()
        stats = graph.build_from_chunks(all_chunks)

        return {
            "status": "graph built",
            "nodes": stats.nodes,
            "edges": stats.edges,
            "components": stats.connected_components,
            "chunks_processed": len(all_chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("graph.build.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))