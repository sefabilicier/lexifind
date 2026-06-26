"""
Output content filter for LexiFind RAG platform.

Validates LLM responses before delivery:
  - Hallucination signal detection (unsupported claims)
  - Sensitive data leakage detection (PII, credentials)
  - Response length sanity check

Reference:
  - OWASP LLM Top 10: LLM02 — Insecure Output Handling
  - IBM Think: output validation in production RAG pipelines
"""

import re
from dataclasses import dataclass

from app.observability.logger import get_logger

logger = get_logger(__name__)

# Patterns that suggest data leakage or hallucination signals
_SENSITIVE_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                            # IP address
    r"(?i)(password|secret|token|api[_\s]?key)\s*[:=]\s*\S+",  # credentials
    r"\b[0-9]{10,}\b",                                          # long number sequences
]

_COMPILED_SENSITIVE = [
    re.compile(p) for p in _SENSITIVE_PATTERNS
]

# Hallucination signal phrases
_HALLUCINATION_SIGNALS = [
    "as of my knowledge cutoff",
    "i don't have access to",
    "i cannot access the internet",
    "based on my training data",
    "i believe but am not certain",
    "you might want to check",
    "i'm not sure but",
]


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_safe: bool
    warnings: list[str]
    filtered_answer: str


class ContentFilter:
    """
    Post-generation output validator.
    Flags responses with sensitive data or hallucination signals.
    """

    def filter(self, answer: str, citations: list[dict]) -> FilterResult:
        """
        Validate and optionally sanitize LLM response.

        Args:
            answer: Raw LLM generated answer.
            citations: Source citations for grounding check.

        Returns:
            FilterResult with safety verdict and cleaned answer.
        """
        warnings = []
        filtered = answer

        # 1. Sensitive data check
        for pattern in _COMPILED_SENSITIVE:
            if pattern.search(answer):
                warnings.append(f"Potential sensitive data detected: {pattern.pattern[:30]}")
                logger.warning(
                    "security.output.sensitive_data",
                    pattern=pattern.pattern[:30],
                )

        # 2. Hallucination signal check
        answer_lower = answer.lower()
        for signal in _HALLUCINATION_SIGNALS:
            if signal in answer_lower:
                warnings.append(f"Hallucination signal: '{signal}'")
                logger.warning(
                    "security.output.hallucination_signal",
                    signal=signal,
                )

        # 3. Citation grounding check
        if not citations:
            warnings.append("Response has no citations — may be ungrounded")
            logger.warning("security.output.no_citations")

        # 4. Minimum length sanity check
        if len(answer.strip()) < 20:
            warnings.append("Response suspiciously short")

        is_safe = len(warnings) == 0

        if not is_safe:
            logger.info(
                "security.output.warnings",
                warning_count=len(warnings),
                warnings=warnings,
            )

        return FilterResult(
            is_safe=is_safe,
            warnings=warnings,
            filtered_answer=filtered,
        )