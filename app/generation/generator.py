"""
Groq LLM wrapper for response generation.

Uses OpenAI-compatible Groq API with:
  - Primary model: llama-3.3-70b-versatile (quality)
  - Fast model:    llama-3.1-8b-instant (classifier, rewrite)

Implements retry/backoff for rate limit handling.
"""

from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# System prompt for legal RAG — strict grounding
_SYSTEM_PROMPT = """You are LexiFind, an expert legal document analysis assistant.

STRICT RULES:
1. Answer ONLY based on the provided context chunks.
2. If the answer is not in the context, say: "I cannot find this information in the provided documents."
3. Always cite your sources using [Source: filename, chunk_index].
4. Be precise — legal documents require exact language.
5. Never hallucinate facts, article numbers, or legal references.

Response format:
- Direct answer first
- Supporting evidence from context
- Source citations at the end
"""


class GroqGenerator:
    """
    Singleton LLM generator using Groq API.
    Supports primary (70B) and fast (8B) model switching.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.client = Groq(api_key=settings.groq_api_key)
        self._initialized = True
        logger.info("generator.ready", model=settings.groq_primary_model)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        use_fast_model: bool = False,
    ) -> dict:
        """
        Generate a grounded response from query + retrieved context.

        Args:
            query: User's original question.
            context_chunks: List of {"text": ..., "metadata": ...} dicts.
            use_fast_model: If True, uses 8B model (faster, cheaper).

        Returns:
            Dict with 'answer', 'model', 'usage' keys.
        """
        model = (
            settings.groq_fast_model
            if use_fast_model
            else settings.groq_primary_model
        )

        # Build context string with source tags
        context_str = self._build_context(context_chunks)

        user_message = f"""CONTEXT:
{context_str}

QUESTION: {query}

Provide a precise answer based solely on the context above."""

        logger.info(
            "generation.started",
            model=model,
            context_chunks=len(context_chunks),
            query=query[:80],
        )

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,   # Low temp for factual legal answers
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        logger.info(
            "generation.completed",
            model=model,
            total_tokens=usage["total_tokens"],
        )

        return {
            "answer": answer,
            "model": model,
            "usage": usage,
        }

    def _build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", "?")
            parts.append(
                f"[{i}] Source: {source} | Chunk: {chunk_idx}\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite user query for better retrieval using fast model.
        Used in Advanced RAG pipeline.

        Transforms conversational queries into precise retrieval queries.
        """
        response = self.client.chat.completions.create(
            model=settings.groq_fast_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query optimization expert for legal document retrieval. "
                        "Rewrite the user's query to maximize retrieval precision. "
                        "Output ONLY the rewritten query, nothing else."
                        "CRITICAL: Output ONLY a natural language search query. "
                        "Never output SQL, code, or structured queries. "
                        "Maximum 15 words. Plain English only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}\nRewritten query:",
                },
            ],
            temperature=0.0,
            max_tokens=128,
        )

        rewritten = response.choices[0].message.content.strip()
        logger.info(
            "query.rewritten",
            original=query[:60],
            rewritten=rewritten[:60],
        )
        return rewritten