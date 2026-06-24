"""
Fixed-size token-based chunker.
Splits text by character count with configurable overlap.
Fastest strategy — good for uniform documents.
"""

from app.ingestion.chunkers.base import BaseChunker, Chunk
from app.config import get_settings


class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size chunks with overlap.
    Uses character count as a proxy for token count (1 token ≈ 4 chars).
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size * 4  # chars
        self.overlap = overlap or settings.chunk_overlap * 4

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "chunker": "fixed_size"},
                    chunk_index=idx,
                ))
                idx += 1

            start += self.chunk_size - self.overlap

        return chunks