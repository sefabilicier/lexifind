"""
Structured logger for LexiFind RAG platform.
Outputs JSON lines in production, pretty-printed in development.
Follows the log format specified in the challenge requirements.
"""

import logging
import sys
from enum import Enum

import structlog
from structlog.types import EventDict

from app.config import get_settings


class EventType(str, Enum):
    QUERY_RECEIVED = "query.received"
    QUERY_CLASSIFIED = "query.classified"
    RETRIEVAL_STARTED = "retrieval.started"
    RETRIEVAL_COMPLETED = "retrieval.completed"
    RERANKING_COMPLETED = "reranking.completed"
    GENERATION_STARTED = "generation.started"
    GENERATION_COMPLETED = "generation.completed"
    FAITHFULNESS_CHECK = "faithfulness.check"
    RESPONSE_DELIVERED = "response.delivered"
    SECURITY_VIOLATION = "security.violation"
    INGESTION_STARTED = "ingestion.started"
    INGESTION_COMPLETED = "ingestion.completed"
    INGESTION_ERROR = "ingestion.error"


def _add_log_level(logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to the event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging() -> None:
    """
    Configure structlog based on LOG_FORMAT env variable.
    json  → JSON lines (production)
    pretty → colored console (development)
    """
    settings = get_settings()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_log_level,
    ]

    if settings.log_format == "pretty":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """Return a bound structlog logger."""
    return structlog.get_logger(name)