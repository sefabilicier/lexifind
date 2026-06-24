"""
Ingestion pipeline orchestrator.
Wires together: Parser → Chunker → Embedder → Qdrant upsert.
"""

from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
)

from app.config import get_settings
from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.html_parser import HTMLParser
from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.chunkers.recursive import RecursiveChunker
from app.ingestion.embedder import BGEEmbedder
from app.observability.logger import get_logger, EventType

logger = get_logger(__name__)
settings = get_settings()

# Extension → Parser mapping
PARSER_REGISTRY: dict[str, BaseParser] = {
    ext: parser
    for parser in [PDFParser(), DOCXParser(), HTMLParser()]
    for ext in parser.supported_extensions
}


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Flow:
        file → Parser → Chunks → BGE-M3 Embeddings → Qdrant
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder: BGEEmbedder | None = None,
        chunker: BaseChunker | None = None,
    ):
        self.client = qdrant_client
        self.embedder = embedder or BGEEmbedder()
        self.chunker = chunker or RecursiveChunker()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not exist."""
        existing = [c.name for c in self.client.get_collections().collections]

        if settings.qdrant_collection not in existing:
            self.client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(
    "ingestion.collection.created",
    collection=settings.qdrant_collection,
)

    def ingest(self, file_path: Path) -> dict:
        """
        Ingest a single document file into Qdrant.

        Args:
            file_path: Path to the document.

        Returns:
            Summary dict with chunk count and point IDs.
        """
        ext = file_path.suffix.lower()
        parser = PARSER_REGISTRY.get(ext)

        if not parser:
            raise ValueError(f"No parser registered for extension: {ext}")

        logger.info(
    "ingestion.started",
    file=file_path.name,
)

        # 1. Parse
        parsed = parser.parse(file_path)

        # 2. Chunk
        chunks = self.chunker.chunk(parsed.content, metadata=parsed.metadata)
        logger.info(
    "ingestion.chunked",
    file=file_path.name,
    chunk_count=len(chunks),
)

        # 3. Embed (dense + sparse in one pass)
        texts = [c.text for c in chunks]
        dense_vecs, sparse_weights = self.embedder.embed_texts(
            texts, return_sparse=True
        )

        # 4. Build Qdrant points
        points = []
        for i, (chunk, dense, sparse) in enumerate(
            zip(chunks, dense_vecs, sparse_weights)
        ):
            # Convert sparse dict {token_id: weight} → SparseVector
            if sparse:
                indices = [int(k) for k in sparse.keys()]
                values = [float(v) for v in sparse.values()]
            else:
                indices, values = [], []

            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        "dense": dense.tolist(),
                        "sparse": SparseVector(
                            indices=indices,
                            values=values,
                        ),
                    },
                    payload={
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
            )

        # 5. Upsert to Qdrant in batches of 64
        batch_size = 64
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[i:i + batch_size],
            )

        logger.info(
    "ingestion.completed",
    file=file_path.name,
    chunks_upserted=len(points),
)

        return {
            "file": file_path.name,
            "chunks": len(points),
            "point_ids": [p.id for p in points],
        }