"""
RAGAS-based evaluation framework for LexiFind RAG pipelines.

Measures four core RAG quality dimensions:

  1. Faithfulness       — Is the answer grounded in retrieved context?
                          Hallucination detection metric.
                          Score: 0.0 (hallucinated) → 1.0 (fully grounded)

  2. Answer Relevancy   — Does the answer address the question?
                          Score: 0.0 (irrelevant) → 1.0 (highly relevant)

  3. Context Precision  — Are retrieved chunks actually useful?
                          Signal/noise ratio of retrieval.
                          Score: 0.0 (noisy) → 1.0 (precise)

  4. Context Recall     — Does retrieved context cover ground truth?
                          Coverage metric.
                          Score: 0.0 (missing info) → 1.0 (full coverage)

Reference:
  - Es et al. (2023) — "RAGAS: Automated Evaluation of RAG Pipelines"
  - https://arxiv.org/abs/2309.15217
  - IBM Think: RAG evaluation as continuous quality gate
"""

from dataclasses import dataclass, field
from groq import Groq
import json

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RAGASScores:
    """RAGAS evaluation scores for a single query."""
    question: str
    answer: str
    ground_truth: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    @property
    def overall(self) -> float:
        """Harmonic mean of all four metrics."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        if all(s == 0 for s in scores):
            return 0.0
        return round(sum(scores) / len(scores), 4)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all golden set questions."""
    total_questions: int = 0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_overall: float = 0.0
    per_question: list[dict] = field(default_factory=list)
    per_category: dict[str, float] = field(default_factory=dict)


# ── LLM-based RAGAS metric prompts ────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are evaluating RAG system faithfulness.

QUESTION: {question}
CONTEXT (retrieved chunks):
{context}
ANSWER: {answer}

Task: Check if every claim in the ANSWER is supported by the CONTEXT.

Respond ONLY with JSON:
{{"score": 0.0-1.0, "reason": "brief explanation"}}

Score guide:
1.0 = all claims in answer are supported by context
0.5 = some claims supported, some not
0.0 = answer contradicts or ignores context
"""

_ANSWER_RELEVANCY_PROMPT = """You are evaluating RAG answer relevancy.

QUESTION: {question}
ANSWER: {answer}

Task: How well does the ANSWER address the QUESTION?

Respond ONLY with JSON:
{{"score": 0.0-1.0, "reason": "brief explanation"}}

Score guide:
1.0 = answer directly and completely addresses the question
0.5 = answer partially addresses the question
0.0 = answer is off-topic or does not address the question
"""

_CONTEXT_PRECISION_PROMPT = """You are evaluating RAG context precision.

QUESTION: {question}
CONTEXT (retrieved chunks):
{context}

Task: What fraction of the retrieved context chunks are actually relevant to answering the question?

Respond ONLY with JSON:
{{"score": 0.0-1.0, "reason": "brief explanation"}}

Score guide:
1.0 = all retrieved chunks are relevant
0.5 = half the chunks are relevant
0.0 = no chunks are relevant to the question
"""

_CONTEXT_RECALL_PROMPT = """You are evaluating RAG context recall.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
CONTEXT (retrieved chunks):
{context}

Task: Does the retrieved context contain the information needed to produce the ground truth answer?

Respond ONLY with JSON:
{{"score": 0.0-1.0, "reason": "brief explanation"}}

Score guide:
1.0 = context fully covers the ground truth
0.5 = context partially covers the ground truth
0.0 = context missing critical information from ground truth
"""


class RAGASEvaluator:
    """
    LLM-as-judge implementation of RAGAS metrics.

    Uses Groq fast model (8B) for cost-efficient evaluation.
    Each metric is scored independently via dedicated prompts.
    """

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    def _score(self, prompt: str) -> tuple[float, str]:
        """
        Run a single evaluation prompt and parse the score.

        Returns:
            Tuple of (score: float, reason: str)
        """
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_fast_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=150,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)
            score = max(0.0, min(1.0, float(result.get("score", 0.0))))
            reason = result.get("reason", "")
            return score, reason

        except Exception as e:
            logger.warning("evaluator.score.failed", error=str(e))
            return 0.0, f"evaluation error: {str(e)}"

    def _format_context(self, context_chunks: list[str]) -> str:
        """Format context chunks for prompt injection."""
        return "\n\n---\n\n".join(
            f"[Chunk {i+1}]: {chunk[:600]}"
            for i, chunk in enumerate(context_chunks)
        )

    def evaluate_single(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context_chunks: list[str],
    ) -> RAGASScores:
        """
        Evaluate a single question-answer pair across all four metrics.

        Args:
            question: Original user question.
            answer: Generated RAG answer.
            ground_truth: Expected correct answer.
            context_chunks: List of retrieved chunk texts.

        Returns:
            RAGASScores with all four metric scores.
        """
        context_str = self._format_context(context_chunks)

        logger.info("evaluator.scoring", question=question[:60])

        # Score all four metrics
        faithfulness, f_reason = self._score(
            _FAITHFULNESS_PROMPT.format(
                question=question,
                context=context_str,
                answer=answer,
            )
        )

        answer_relevancy, ar_reason = self._score(
            _ANSWER_RELEVANCY_PROMPT.format(
                question=question,
                answer=answer,
            )
        )

        context_precision, cp_reason = self._score(
            _CONTEXT_PRECISION_PROMPT.format(
                question=question,
                context=context_str,
            )
        )

        context_recall, cr_reason = self._score(
            _CONTEXT_RECALL_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                context=context_str,
            )
        )

        scores = RAGASScores(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
        )

        logger.info(
            "evaluator.scored",
            question=question[:60],
            faithfulness=scores.faithfulness,
            answer_relevancy=scores.answer_relevancy,
            context_precision=scores.context_precision,
            context_recall=scores.context_recall,
            overall=scores.overall,
        )

        return scores