"""
Community detection using Louvain algorithm.

Communities = clusters of tightly related legal entities.
Used in Graph RAG to build community summaries for global context.

Reference:
  - Blondel et al. (2008) — Louvain method for community detection
  - Microsoft GraphRAG: community summaries as RAG context
"""

import community as community_louvain
import networkx as nx

from app.observability.logger import get_logger

logger = get_logger(__name__)


class CommunityDetector:
    """
    Detects entity communities in the knowledge graph using Louvain.
    Each community represents a thematic cluster of legal concepts.
    """

    def detect(self, graph: nx.DiGraph) -> dict[int, list[str]]:
        """
        Detect communities in the graph.

        Args:
            graph: Directed knowledge graph.

        Returns:
            Dict mapping community_id → list of entity node IDs.
        """
        if graph.number_of_nodes() == 0:
            return {}

        # Louvain works on undirected graphs
        undirected = graph.to_undirected()
        partition: dict[str, int] = community_louvain.best_partition(undirected)

        # Invert: community_id → [node_ids]
        communities: dict[int, list[str]] = {}
        for node_id, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node_id)

        logger.info(
            "graph.communities.detected",
            total_communities=len(communities),
            total_nodes=graph.number_of_nodes(),
        )

        return communities

    def summarize_community(
        self,
        community_nodes: list[str],
        graph: nx.DiGraph,
    ) -> str:
        """
        Build a concise text summary for a community.
        Used as additional context in Graph RAG prompts.
        """
        labels = [
            graph.nodes[n].get("label", n)
            for n in community_nodes
            if n in graph
        ]
        types = [
            graph.nodes[n].get("type", "?")
            for n in community_nodes
            if n in graph
        ]

        type_groups: dict[str, list[str]] = {}
        for label, etype in zip(labels, types):
            type_groups.setdefault(etype, []).append(label)

        parts = []
        for etype, group_labels in type_groups.items():
            parts.append(f"{etype}: {', '.join(group_labels)}")

        return " | ".join(parts)