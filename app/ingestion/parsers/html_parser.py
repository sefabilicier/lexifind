"""
HTML parser using BeautifulSoup4.
Strips scripts/styles and extracts clean readable text.
"""

from pathlib import Path

from bs4 import BeautifulSoup

from app.ingestion.parsers.base import BaseParser, ParsedDocument
from app.observability.logger import get_logger

logger = get_logger(__name__)

_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside"]


class HTMLParser(BaseParser):

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def parse(self, file_path: Path) -> ParsedDocument:
        logger.info(
            "Parsing HTML",
            file=file_path.name,
        )

        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")

        # Remove noise tags
        for tag in soup(tags := _NOISE_TAGS):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else ""
        full_content = soup.get_text(separator="\n", strip=True)

        metadata = {
            "source": file_path.name,
            "file_type": "html",
            "title": title,
        }

        logger.info(
            "HTML parsed successfully",
            file=file_path.name,
            chars=len(full_content),
        )

        return ParsedDocument(
            content=full_content,
            metadata=metadata,
            pages=[full_content],
        )