"""
API Key authentication middleware.
Validates X-API-Key header against the in-memory generated key.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.security.api_key_manager import api_key_manager
from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_PUBLIC_PATHS = {
    "/api/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header on all protected endpoints.
    Key is resolved from in-memory APIKeyManager — no .env lookup.
    """

    async def dispatch(self, request: Request, call_next):
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

        if not api_key_manager.is_valid(api_key):
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