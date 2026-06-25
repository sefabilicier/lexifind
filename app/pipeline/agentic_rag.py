"""
Agentic RAG Pipeline using LangGraph.

Agent loop:
  1. Plan     — decompose complex query into sub-questions
  2. Retrieve — hybrid search for each sub-question
  3. Evaluate — are retrieved chunks sufficient?
     - YES → Generate final answer
     - NO  → Refine sub-questions and retry (max iterations)

This handles multi-hop reasoning that single-pass RAG cannot.

Reference:
  - ReAct (Yao et al. 2022): Reasoning + Acting in LLMs
  - LangGraph: stateful multi-step agent graphs
  - IBM Think: agentic RAG for complex document QA
"""

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.embedder import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import BGEReranker
from app.generation.generator import GroqGenerator
from app.generation.citation_builder import build_citations
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── LangGraph State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between LangGraph nodes."""
    original_query: str
    sub_questions: list[str]
    retrieved_chunks: Annotated[list, operator.add]   # accumulates across nodes
    reranked_chunks: list
    answer: str
    citations: list[dict]
    usage: dict
    iteration: int
    sufficient: bool


# ── Node functions ─────────────────────────────────────────────────────────────

def plan_node(state: AgentState, generator: GroqGenerator) -> AgentState:
    """
    Decompose the original query into focused sub-questions.
    Uses fast 8B model to save tokens.
    """
    query = state["original_query"]

    response = generator.client.chat.completions.create(
        model=settings.groq_fast_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal document analyst. "
                    "Decompose the question into 2-3 precise sub-questions "
                    "for targeted retrieval. Output ONLY a numbered list, nothing else."
                ),
            },
            {"role": "user", "content": f"Question: {query}"},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    sub_questions = [
        line.lstrip("0123456789.-) ").strip()
        for line in raw.split("\n")
        if line.strip()
    ]

    logger.info(
        "agent.plan",
        sub_questions=sub_questions,
        iteration=state["iteration"],
    )

    return {**state, "sub_questions": sub_questions}


def retrieve_node(
    state: AgentState,
    retriever: HybridRetriever,
    top_k: int,
) -> AgentState:
    """
    Retrieve chunks for each sub-question independently.
    Results accumulate via Annotated[list, operator.add].
    """
    all_chunks = []
    for sub_q in state["sub_questions"]:
        chunks = retriever.retrieve(sub_q, top_k=top_k)
        all_chunks.extend(chunks)

    logger.info(
        "agent.retrieve",
        sub_questions=len(state["sub_questions"]),
        chunks_found=len(all_chunks),
    )

    return {**state, "retrieved_chunks": all_chunks}


def evaluate_node(
    state: AgentState,
    reranker: BGEReranker,
    top_n: int,
    threshold: float = 0.3,
) -> AgentState:
    """
    Evaluate retrieval sufficiency via reranker scores.
    If top chunk score < threshold → insufficient → retry.
    """
    # Deduplicate by chunk ID
    seen, unique = set(), []
    for chunk in state["retrieved_chunks"]:
        if chunk.id not in seen:
            seen.add(chunk.id)
            unique.append(chunk)

    reranked = reranker.rerank(
        state["original_query"], unique, top_n=top_n
    )

    sufficient = bool(reranked and reranked[0].score >= threshold)

    logger.info(
        "agent.evaluate",
        top_score=round(reranked[0].score, 4) if reranked else 0,
        sufficient=sufficient,
        iteration=state["iteration"],
    )

    return {
        **state,
        "reranked_chunks": reranked,
        "sufficient": sufficient,
        "iteration": state["iteration"] + 1,
    }


def generate_node(
    state: AgentState,
    generator: GroqGenerator,
) -> AgentState:
    """Generate final answer from reranked chunks."""
    chunks = state["reranked_chunks"]
    context = [{"text": c.text, "metadata": c.metadata} for c in chunks]
    result = generator.generate(state["original_query"], context)
    citations = build_citations(chunks)

    logger.info(
        "agent.generate",
        total_tokens=result["usage"]["total_tokens"],
    )

    return {
        **state,
        "answer": result["answer"],
        "citations": citations,
        "usage": result["usage"],
    }


def should_continue(state: AgentState) -> str:
    """
    LangGraph conditional edge.
    Routes to 'generate' if sufficient or max iterations reached.
    """
    if state["sufficient"] or state["iteration"] >= settings.max_agent_iterations:
        return "generate"
    return "retrieve"


# ── Pipeline class ─────────────────────────────────────────────────────────────

class AgenticRAGPipeline:
    """
    LangGraph-based agentic RAG with plan → retrieve → evaluate loop.
    """

    def __init__(self, client: QdrantClient):
        embedder = BGEEmbedder()
        self.retriever = HybridRetriever(client=client, embedder=embedder)
        self.reranker = BGEReranker()
        self.generator = GroqGenerator()
        self.graph = self._build_graph()

    def _build_graph(self) -> any:
        """Wire up LangGraph nodes and edges."""
        retriever = self.retriever
        reranker = self.reranker
        generator = self.generator

        builder = StateGraph(AgentState)

        builder.add_node(
            "plan",
            lambda s: plan_node(s, generator),
        )
        builder.add_node(
            "retrieve",
            lambda s: retrieve_node(s, retriever, top_k=settings.top_k_retrieval),
        )
        builder.add_node(
            "evaluate",
            lambda s: evaluate_node(s, reranker, top_n=settings.final_n_rerank),
        )
        builder.add_node(
            "generate",
            lambda s: generate_node(s, generator),
        )

        builder.set_entry_point("plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            should_continue,
            {"retrieve": "retrieve", "generate": "generate"},
        )
        builder.add_edge("generate", END)

        return builder.compile()

    def run(self, query: str, top_k: int = 10, top_n: int = 3) -> dict:
        """Execute agentic RAG pipeline."""
        logger.info("pipeline.agentic.started", query=query[:80])

        initial_state: AgentState = {
            "original_query": query,
            "sub_questions": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "citations": [],
            "usage": {},
            "iteration": 0,
            "sufficient": False,
        }

        final_state = self.graph.invoke(initial_state)

        logger.info(
            "pipeline.agentic.completed",
            iterations=final_state["iteration"],
            chunks_used=len(final_state["reranked_chunks"]),
        )

        return {
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "usage": final_state["usage"],
            "pipeline": "agentic_rag",
            "iterations": final_state["iteration"],
            "sub_questions": final_state["sub_questions"],
            "chunks_used": len(final_state["reranked_chunks"]),
        }