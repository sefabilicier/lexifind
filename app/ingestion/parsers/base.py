"""
Abstract base class for all document parsers.
Every parser must implement the `parse` method and return a ParsedDocument.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    """
    Unified output structure from any parser.

    Attributes:
        content: Full extracted text
        metadata: Source metadata (filename, page count, author, etc.)
        pages: Per-page text if available (PDF, DOCX)
    """
    content: str
    metadata: dict = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)


class BaseParser(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a document file and return structured content.

        Args:
            file_path: Absolute path to the document file.

        Returns:
            ParsedDocument with extracted text and metadata.
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of file extensions this parser handles, e.g. ['.pdf']"""
        ...