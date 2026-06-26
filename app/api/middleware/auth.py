"""
API Key authentication middleware.

Simple but production-ready API key auth:
  - Key passed via X-API-Key header
  - Keys stored in .env (comma-separated for multiple clients)
  - Public endpoints (health, docs) are exempt

For production upgrade path: replace with JWT or OAuth2.

Reference:
  - FastAPI security best practices
  - OWASP API Security Top 10: API2 — Broken Authentication
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Endpoints that do NOT require authentication
_PUBLIC_PATHS = {
    "/api/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header on all protected endpoints.
    Returns 401 for missing key, 403 for invalid key.
    """

    def __init__(self, app, valid_keys: set[str]):
        super().__init__(app)
        self.valid_keys = valid_keys

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get(settings.api_key_header)

        if not api_key:
            logger.warning(
                "security.auth.missing_key",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Include X-API-Key header."},
            )

        if api_key not in self.valid_keys:
            logger.warning(
                "security.auth.invalid_key",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)