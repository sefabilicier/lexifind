"""
PDF parser using PyMuPDF (fitz).
Extracts text per page with metadata (title, author, page count).
"""

from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.parsers.base import BaseParser, ParsedDocument
from app.observability.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        logger.info(
            "Parsing PDF",
            file=file_path.name,
        )

        doc = fitz.open(str(file_path))
        pages = []

        for page in doc:
            text = page.get_text("text")
            pages.append(text.strip())

        full_content = "\n\n".join(p for p in pages if p)

        metadata = {
            "source": file_path.name,
            "file_type": "pdf",
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
        }

        doc.close()

        logger.info(
            "PDF parsed successfully",
            file=file_path.name,
            pages=len(pages),
            chars=len(full_content),
        )

        return ParsedDocument(
            content=full_content,
            metadata=metadata,
            pages=pages,
        )