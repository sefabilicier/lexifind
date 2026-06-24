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