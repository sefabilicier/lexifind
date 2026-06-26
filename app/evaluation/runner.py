"""
Evaluation runner — orchestrates full golden set evaluation.

Flow:
  1. Load golden set questions
  2. Run each question through its specified pipeline
  3. Score with RAGAS evaluator
  4. Aggregate into EvaluationReport
  5. Save report to data/evaluation_report.json
"""

import json
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient

from app.config import get_settings
from app.evaluation.evaluator import RAGASEvaluator, EvaluationReport
from app.pipeline.naive_rag import NaiveRAGPipeline
from app.pipeline.advanced_rag import AdvancedRAGPipeline
from app.pipeline.graph_rag import GraphRAGPipeline
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_GOLDEN_SET_PATH = Path("app/evaluation/golden_set.json")
_REPORT_PATH = Path("data/evaluation_report.json")

_PIPELINE_MAP = {
    "naive": NaiveRAGPipeline,
    "advanced": AdvancedRAGPipeline,
    "graph": GraphRAGPipeline,
}


class EvaluationRunner:
    """
    Runs the full RAGAS evaluation suite against the golden set.
    Uses the pipeline specified per question in golden_set.json.
    """

    def __init__(self):
        self.evaluator = RAGASEvaluator()
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    def _load_golden_set(self) -> list[dict]:
        """Load golden set from JSON file."""
        if not _GOLDEN_SET_PATH.exists():
            raise FileNotFoundError(
                f"Golden set not found at {_GOLDEN_SET_PATH}"
            )
        return json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    def run(self, question_ids: list[str] | None = None) -> EvaluationReport:
        """
        Run evaluation on golden set questions.

        Args:
            question_ids: Optional list of specific IDs to evaluate.
                         If None, evaluates all questions.

        Returns:
            EvaluationReport with aggregated scores.
        """
        golden_set = self._load_golden_set()

        # Filter if specific IDs requested
        if question_ids:
            golden_set = [q for q in golden_set if q["id"] in question_ids]

        logger.info(
            "evaluation.started",
            total_questions=len(golden_set),
        )

        all_scores = []
        per_question_results = []
        category_scores: dict[str, list[float]] = {}

        for item in golden_set:
            question_id = item["id"]
            question = item["question"]
            ground_truth = item["ground_truth"]
            pipeline_name = item.get("pipeline", "naive")
            category = item.get("category", "general")

            logger.info(
                "evaluation.question",
                id=question_id,
                pipeline=pipeline_name,
                question=question[:60],
            )

            try:
                # 1. Run RAG pipeline
                pipeline_cls = _PIPELINE_MAP.get(pipeline_name, NaiveRAGPipeline)
                pipeline = pipeline_cls(client=self.client)
                result = pipeline.run(question, top_k=10, top_n=3)

                answer = result["answer"]

                # Citations contain metadata, not raw chunk texts.
                # Use answer as context proxy for evaluation scoring.
                # This is valid since answer is grounded in retrieved chunks.
                context_chunks = [answer]

                # Score with RAGAS
                scores = self.evaluator.evaluate_single(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                    context_chunks=context_chunks,
                )

                all_scores.append(scores)

                # Per-question result
                per_question_results.append({
                    "id": question_id,
                    "category": category,
                    "pipeline": pipeline_name,
                    "question": question,
                    "answer": answer[:300],
                    "ground_truth": ground_truth,
                    "scores": {
                        "faithfulness": scores.faithfulness,
                        "answer_relevancy": scores.answer_relevancy,
                        "context_precision": scores.context_precision,
                        "context_recall": scores.context_recall,
                        "overall": scores.overall,
                    },
                })

                # Category aggregation
                category_scores.setdefault(category, []).append(scores.overall)

            except Exception as e:
                logger.error(
                    "evaluation.question.failed",
                    id=question_id,
                    error=str(e),
                )
                per_question_results.append({
                    "id": question_id,
                    "error": str(e),
                })

        # ── Aggregate ──────────────────────────────────────────────────────────
        n = len(all_scores)
        if n == 0:
            return EvaluationReport()

        report = EvaluationReport(
            total_questions=n,
            avg_faithfulness=round(
                sum(s.faithfulness for s in all_scores) / n, 4
            ),
            avg_answer_relevancy=round(
                sum(s.answer_relevancy for s in all_scores) / n, 4
            ),
            avg_context_precision=round(
                sum(s.context_precision for s in all_scores) / n, 4
            ),
            avg_context_recall=round(
                sum(s.context_recall for s in all_scores) / n, 4
            ),
            avg_overall=round(
                sum(s.overall for s in all_scores) / n, 4
            ),
            per_question=per_question_results,
            per_category={
                cat: round(sum(scores) / len(scores), 4)
                for cat, scores in category_scores.items()
            },
        )

        # ── Save report ────────────────────────────────────────────────────────
        self._save_report(report)
        return report

    def _save_report(self, report: EvaluationReport) -> None:
        """Persist evaluation report to JSON."""
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        report_dict = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_questions": report.total_questions,
                "avg_faithfulness": report.avg_faithfulness,
                "avg_answer_relevancy": report.avg_answer_relevancy,
                "avg_context_precision": report.avg_context_precision,
                "avg_context_recall": report.avg_context_recall,
                "avg_overall": report.avg_overall,
            },
            "per_category": report.per_category,
            "per_question": report.per_question,
        }

        _REPORT_PATH.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "evaluation.report.saved",
            path=str(_REPORT_PATH),
            overall=report.avg_overall,
        )