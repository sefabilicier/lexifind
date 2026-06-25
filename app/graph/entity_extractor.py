"""
Legal entity extractor using Groq LLM.

Extracts structured entities and relationships from text chunks.
Designed for legal documents — recognizes:
  - LAW: Specific laws, regulations, directives
  - ARTICLE: Article/section references
  - CONCEPT: Legal concepts and technical terms
  - REQUIREMENT: Obligations, constraints, deadlines
  - METRIC: Evaluation metrics, thresholds, KPIs
  - ROLE: Actors, parties, stakeholders

Output is a structured JSON graph fragment:
  {
    "entities": [{"id": str, "label": str, "type": str}],
    "relations": [{"source": str, "target": str, "relation": str}]
  }

Reference:
  - Edge et al. (2024) — "From Local to Global: A Graph RAG Approach"
    Microsoft Research: GraphRAG paper
  - IBM Think: Knowledge graph construction for RAG
"""

import json
from groq import Groq

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_EXTRACTOR_SYSTEM_PROMPT = """You are a legal knowledge graph builder.

Extract entities and relationships from the given text chunk.

Entity types:
  - LAW: specific laws, regulations, acts (e.g. "GDPR", "Turkish Commercial Code")
  - ARTICLE: article or section references (e.g. "Article 5", "Section 2.3")
  - CONCEPT: legal/technical concepts (e.g. "RAG pipeline", "hybrid retrieval")
  - REQUIREMENT: obligations or constraints (e.g. "must implement reranking")
  - METRIC: evaluation metrics or KPIs (e.g. "faithfulness score", "RAGAS")
  - ROLE: actors or parties (e.g. "senior developer", "evaluator")

Relation types (use short verb phrases):
  - requires, defines, references, part_of, evaluates, implements, constrains

Output ONLY valid JSON in this exact format:
{
  "entities": [
    {"id": "unique_snake_case_id", "label": "Display Name", "type": "ENTITY_TYPE"}
  ],
  "relations": [
    {"source": "entity_id_1", "target": "entity_id_2", "relation": "verb_phrase"}
  ]
}

Rules:
- Maximum 8 entities and 8 relations per chunk
- IDs must be snake_case, unique, and descriptive
- Only extract entities clearly present in the text
- Output ONLY the JSON object, no explanation
"""


class EntityExtractor:
    """
    Extracts legal entities and relations from text using Groq fast model.
    Output is compatible with NetworkX graph node/edge format.
    """

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    def extract(self, text: str, chunk_id: str) -> dict:
        """
        Extract entities and relations from a single text chunk.

        Args:
            text: Chunk text content.
            chunk_id: Source chunk identifier for provenance tracking.

        Returns:
            Dict with 'entities' and 'relations' lists.
            Falls back to empty structure on parse failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_fast_model,
                messages=[
                    {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"TEXT CHUNK (id={chunk_id}):\n\n{text[:1200]}",
                    },
                ],
                temperature=0.0,
                max_tokens=512,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)

            # Attach provenance to each entity
            for entity in result.get("entities", []):
                entity["source_chunk"] = chunk_id

            logger.info(
                "graph.extract.completed",
                chunk_id=chunk_id,
                entities=len(result.get("entities", [])),
                relations=len(result.get("relations", [])),
            )

            return result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning(
                "graph.extract.failed",
                chunk_id=chunk_id,
                error=str(e),
            )
            return {"entities": [], "relations": []}

    def extract_batch(
        self, chunks: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Extract entities and relations from multiple chunks.

        Args:
            chunks: List of {"id": str, "text": str} dicts.

        Returns:
            Tuple of (all_entities, all_relations).
        """
        all_entities, all_relations = [], []

        for chunk in chunks:
            result = self.extract(chunk["text"], chunk["id"])
            all_entities.extend(result.get("entities", []))
            all_relations.extend(result.get("relations", []))

        logger.info(
            "graph.extract_batch.completed",
            chunks=len(chunks),
            total_entities=len(all_entities),
            total_relations=len(all_relations),
        )

        return all_entities, all_relations