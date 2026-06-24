"""
Abstract base class for all chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    A single text chunk with its metadata.

    Attributes:
        text: Chunk content
        metadata: Inherited + chunk-specific metadata
        chunk_index: Position in the original document
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


class BaseChunker(ABC):
    """Abstract interface for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Input text to split.
            metadata: Document-level metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        ...