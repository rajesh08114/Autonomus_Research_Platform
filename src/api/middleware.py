from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.core.logger import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = f"req_{uuid.uuid4().hex[:12]}"
        start = time.time()
        logger.info(
            "http.request.start",
            request_id=request.state.request_id,
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http.request.error",
                request_id=request.state.request_id,
                method=request.method,
                path=request.url.path,
            )
            raise
        duration = time.time() - start
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        logger.info(
            "http.request.end",
            request_id=request.state.request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response


def register_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)
