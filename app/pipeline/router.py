"""
Query Router — classifies incoming queries and selects the optimal pipeline.

Pipeline selection logic:
  - simple    → Naive RAG   (single-hop, factual lookup)
  - complex   → Advanced RAG (multi-concept, needs rewriting)
  - multi_hop → Agentic RAG  (requires reasoning across chunks)
  - uncertain → CRAG         (low-confidence retrieval)

Uses Llama-3.1-8B (fast model) for classification — no wasted 70B tokens.

Reference:
  - RouteLLM (2024): Routing queries to appropriate model/pipeline
  - IBM Think RAG: query complexity classification as first step
"""

from groq import Groq

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_ROUTER_SYSTEM_PROMPT = """You are a query complexity classifier for a legal RAG system.

Classify the query into exactly ONE of these categories:

- simple: Single-hop factual question. Answer found in one passage.
  Example: "What is the deadline for submission?"

- complex: Multi-concept question needing query rewriting and broader search.
  Example: "How do evaluation metrics relate to pipeline performance?"

- multi_hop: Requires reasoning across multiple documents or steps.
  Example: "Compare the requirements in section 2 with the constraints in section 5."

- uncertain: Query is ambiguous, vague, or likely to produce low-confidence retrieval.
  Example: "Tell me about the stuff mentioned earlier."

Respond with ONLY one word: simple | complex | multi_hop | uncertain
"""


class QueryRouter:
    """
    Classifies queries to route them to the most appropriate RAG pipeline.
    Uses fast 8B model to minimize latency overhead.
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

    def classify(self, query: str) -> str:
        """
        Classify a query into a pipeline category.

        Returns:
            One of: 'simple' | 'complex' | 'multi_hop' | 'uncertain'
        """
        response = self.client.chat.completions.create(
            model=settings.groq_fast_model,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        label = response.choices[0].message.content.strip().lower()

        # Sanitize — fallback to 'complex' if unexpected output
        valid = {"simple", "complex", "multi_hop", "uncertain"}
        if label not in valid:
            label = "complex"

        logger.info("router.classified", query=query[:60], label=label)
        return label

    def route(self, query: str) -> str:
        """
        Map classification label to pipeline name.

        Returns:
            Pipeline name: 'naive' | 'advanced' | 'agentic' | 'corrective'
        """
        label = self.classify(query)
        mapping = {
            "simple": "naive",
            "complex": "advanced",
            "multi_hop": "agentic",
            "uncertain": "corrective",
        }
        pipeline = mapping[label]
        logger.info("router.routed", label=label, pipeline=pipeline)
        return pipeline