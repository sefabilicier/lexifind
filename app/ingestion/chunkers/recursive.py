"""
Recursive character splitter.
Tries to split on paragraph → sentence → word boundaries before hard-cutting.
Best general-purpose chunker for legal documents.
"""

import re
from app.ingestion.chunkers.base import BaseChunker, Chunk
from app.config import get_settings

# Priority order: split on these separators before hard-cutting
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.
    Preserves semantic boundaries (paragraphs → sentences → words).
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size * 4
        self.overlap = overlap or settings.chunk_overlap * 4
        self.separators = separators or _SEPARATORS

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            # Last resort: hard character cut
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        good, current = [], ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good.append(current)
                if len(part) > self.chunk_size:
                    good.extend(self._split(part, remaining))
                    current = ""
                else:
                    current = part

        if current:
            good.append(current)

        return good

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        raw_chunks = self._split(text, self.separators)
        chunks = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Add overlap from previous chunk
            if idx > 0 and self.overlap > 0:
                prev = chunks[-1].text if chunks else ""
                overlap_text = prev[-self.overlap:]
                chunk_text = overlap_text + " " + chunk_text

            chunks.append(Chunk(
                text=chunk_text.strip(),
                metadata={**metadata, "chunker": "recursive"},
                chunk_index=idx,
            ))

        return chunks