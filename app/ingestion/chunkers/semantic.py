"""
Semantic chunker using embedding similarity.
Groups sentences into chunks based on cosine similarity breakpoints.
Best for: heterogeneous documents where topic shifts matter.
Reference: Greg Kamradt's semantic chunking approach.
"""

import re
import numpy as np
from app.ingestion.chunkers.base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """
    Splits text at semantic breakpoints detected via embedding similarity drops.
    Requires an embedder to be injected (lazy — avoids circular imports).
    """

    def __init__(
        self,
        embedder,
        breakpoint_threshold: float = 0.3,
        min_chunk_size: int = 100,
    ):
        """
        Args:
            embedder: Any object with an `embed_texts(List[str]) -> np.ndarray` method.
            breakpoint_threshold: Cosine distance above this = new chunk boundary.
            min_chunk_size: Minimum characters per chunk.
        """
        self.embedder = embedder
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return [Chunk(text=text, metadata=metadata, chunk_index=0)]

        # Embed all sentences in one batch
        embeddings = self.embedder.embed_texts(sentences)

        # Compute cosine distances between consecutive sentences
        def cosine_distance(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        distances = [
            cosine_distance(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Find breakpoints where distance exceeds threshold
        breakpoints = {
            i + 1
            for i, d in enumerate(distances)
            if d > self.breakpoint_threshold
        }

        # Build chunks from breakpoints
        chunks, current_sentences, idx = [], [], 0
        for i, sentence in enumerate(sentences):
            if i in breakpoints and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**metadata, "chunker": "semantic"},
                        chunk_index=idx,
                    ))
                    idx += 1
                current_sentences = []
            current_sentences.append(sentence)

        # Last group
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "chunker": "semantic"},
                    chunk_index=idx,
                ))

        return chunks