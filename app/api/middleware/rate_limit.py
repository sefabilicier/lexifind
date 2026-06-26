"""
Rate limiting middleware using slowapi (Starlette-compatible).

Limits:
  - Per-minute: prevents burst abuse
  - Applied per client IP

Reference:
  - OWASP API Security: API4 — Lack of Resources & Rate Limiting
  - FastAPI + slowapi production pattern
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global limiter instance — imported by routers
limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded errors."""
    logger.warning(
        "security.rate_limit.exceeded",
        client=request.client.host if request.client else "unknown",
        path=request.url.path,
        limit=str(exc.detail),
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down.",
            "retry_after": "60 seconds",
        },
    )