"""
Knowledge graph builder using NetworkX.

Constructs an in-memory directed graph from extracted entities and relations.
Each node = legal entity, each edge = named relation.

Graph is used for:
  - Community-based context expansion (Graph RAG)
  - Entity-centric retrieval (find all chunks related to an entity)
  - Relationship traversal (multi-hop reasoning support)

Reference:
  - Edge et al. (2024) Microsoft GraphRAG
  - NetworkX: industry-standard Python graph library
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

import networkx as nx

from app.graph.entity_extractor import EntityExtractor
from app.observability.logger import get_logger

logger = get_logger(__name__)

_GRAPH_CACHE_PATH = Path("data/graph_cache.json")


@dataclass
class GraphStats:
    nodes: int = 0
    edges: int = 0
    communities: int = 0
    connected_components: int = 0


class LegalKnowledgeGraph:
    """
    In-memory directed knowledge graph for legal documents.

    Nodes: entities with type, label, source_chunk attributes
    Edges: directed relations with relation label

    Persists to JSON for reuse across server restarts.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.extractor = EntityExtractor()
        _GRAPH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def build_from_chunks(self, chunks: list[dict]) -> GraphStats:
        """
        Build graph from a list of text chunks.

        Args:
            chunks: List of {"id": str, "text": str} dicts.

        Returns:
            GraphStats summary.
        """
        logger.info("graph.build.started", chunk_count=len(chunks))

        entities, relations = self.extractor.extract_batch(chunks)

        # Add nodes
        for entity in entities:
            node_id = entity["id"]
            self.graph.add_node(
                node_id,
                label=entity.get("label", node_id),
                type=entity.get("type", "UNKNOWN"),
                source_chunk=entity.get("source_chunk", ""),
            )

        # Add edges — only between known nodes
        for rel in relations:
            src, tgt = rel.get("source"), rel.get("target")
            if src in self.graph and tgt in self.graph:
                self.graph.add_edge(
                    src,
                    tgt,
                    relation=rel.get("relation", "related_to"),
                )

        stats = GraphStats(
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
            connected_components=nx.number_weakly_connected_components(self.graph),
        )

        logger.info(
            "graph.build.completed",
            nodes=stats.nodes,
            edges=stats.edges,
            components=stats.connected_components,
        )

        self._save_cache()
        return stats

    def get_neighbors(self, entity_id: str, depth: int = 2) -> list[str]:
        """
        Return all entity IDs reachable from entity_id within `depth` hops.
        Used for context expansion in Graph RAG.
        """
        if entity_id not in self.graph:
            return []

        neighbors = set()
        frontier = {entity_id}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            neighbors.update(next_frontier)
            frontier = next_frontier

        neighbors.discard(entity_id)
        return list(neighbors)

    def get_entity_chunks(self, entity_ids: list[str]) -> list[str]:
        """Return source chunk IDs for given entity IDs."""
        chunks = set()
        for eid in entity_ids:
            if eid in self.graph:
                chunk_id = self.graph.nodes[eid].get("source_chunk", "")
                if chunk_id:
                    chunks.add(chunk_id)
        return list(chunks)


    def find_entities_in_text(self, text: str) -> list[str]:
        """
        Find graph entities mentioned in a query using flexible matching.
        Checks both label and individual words for partial matches.
        """
        text_lower = text.lower()
        text_words = set(text_lower.split())
        matches = []

        for node_id, data in self.graph.nodes(data=True):
            label = data.get("label", "").lower()
            if not label:
                continue

            # Exact label match
            if label in text_lower:
                matches.append(node_id)
                continue

            # Partial: any word of the label found in query
            label_words = set(label.split())
            if label_words & text_words:  # intersection
                matches.append(node_id)

        logger.info(
            "graph.entity_match",
            query=text[:60],
            matched=matches[:5],
        )
        return matches
    

    def get_subgraph_summary(self, entity_ids: list[str]) -> str:
        """
        Generate a human-readable summary of a subgraph.
        Passed to LLM as additional graph context.
        """
        if not entity_ids:
            return ""

        lines = ["Knowledge Graph Context:"]
        for eid in entity_ids:
            if eid not in self.graph:
                continue
            data = self.graph.nodes[eid]
            lines.append(
                f"  - [{data.get('type', '?')}] {data.get('label', eid)}"
            )
            for _, tgt, edge_data in self.graph.out_edges(eid, data=True):
                tgt_label = self.graph.nodes[tgt].get("label", tgt)
                lines.append(
                    f"      → {edge_data.get('relation', 'related_to')} → {tgt_label}"
                )

        return "\n".join(lines)

    def _save_cache(self) -> None:
        """Persist graph to JSON for reuse across restarts."""
        data = nx.node_link_data(self.graph)
        _GRAPH_CACHE_PATH.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        logger.info("graph.cache.saved", path=str(_GRAPH_CACHE_PATH))

    def load_cache(self) -> bool:
        """
        Load graph from JSON cache if it exists.
        Returns True if loaded successfully.
        """
        if not _GRAPH_CACHE_PATH.exists():
            return False
        data = json.loads(_GRAPH_CACHE_PATH.read_text(encoding="utf-8"))
        self.graph = nx.node_link_graph(data)
        logger.info(
            "graph.cache.loaded",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )
        return True