"""
Evaluation API endpoints.

POST /api/evaluate          → Run full golden set evaluation
POST /api/evaluate/single   → Evaluate a single custom question
GET  /api/evaluate/report   → Retrieve latest evaluation report
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.evaluation.runner import EvaluationRunner, _REPORT_PATH
from app.evaluation.evaluator import RAGASEvaluator
from app.api.middleware.rate_limit import limiter
from app.config import get_settings
from app.observability.logger import get_logger

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])
logger = get_logger(__name__)
settings = get_settings()


class SingleEvalRequest(BaseModel):
    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=5)
    ground_truth: str = Field(..., min_length=5)
    context_chunks: list[str] = Field(default_factory=list)


class RunEvalRequest(BaseModel):
    question_ids: list[str] | None = Field(
        default=None,
        description="Specific question IDs to evaluate. None = all."
    )


@router.post("/")
@limiter.limit("5/minute")
async def run_evaluation(request: Request, body: RunEvalRequest):
    """
    Run RAGAS evaluation on golden set.
    Warning: This makes multiple LLM calls — may take 2-5 minutes.
    """
    try:
        logger.info(
            "evaluation.run.started",
            question_ids=body.question_ids,
        )

        runner = EvaluationRunner()
        report = runner.run(question_ids=body.question_ids)

        return {
            "status": "completed",
            "summary": {
                "total_questions": report.total_questions,
                "avg_faithfulness": report.avg_faithfulness,
                "avg_answer_relevancy": report.avg_answer_relevancy,
                "avg_context_precision": report.avg_context_precision,
                "avg_context_recall": report.avg_context_recall,
                "avg_overall": report.avg_overall,
            },
            "per_category": report.per_category,
            "per_question_count": len(report.per_question),
        }

    except Exception as e:
        logger.error("evaluation.run.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/single")
@limiter.limit("10/minute")
async def evaluate_single(request: Request, body: SingleEvalRequest):
    """
    Evaluate a single question-answer pair with RAGAS metrics.
    Useful for spot-checking specific pipeline outputs.
    """
    try:
        evaluator = RAGASEvaluator()
        scores = evaluator.evaluate_single(
            question=body.question,
            answer=body.answer,
            ground_truth=body.ground_truth,
            context_chunks=body.context_chunks,
        )

        return {
            "faithfulness": scores.faithfulness,
            "answer_relevancy": scores.answer_relevancy,
            "context_precision": scores.context_precision,
            "context_recall": scores.context_recall,
            "overall": scores.overall,
        }

    except Exception as e:
        logger.error("evaluation.single.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_report():
    """
    Retrieve the latest saved evaluation report.
    """
    if not _REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found. Run POST /api/evaluate first.",
        )

    report = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    return report