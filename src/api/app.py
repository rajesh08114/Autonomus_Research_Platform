from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import register_middleware
from src.api.router import api_router
from src.config.settings import settings
from src.db.database import init_db
from src.core.logging import configure_logging
from src.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    logger.info("app.startup", env=settings.APP_ENV, version="2.0.0")
    await init_db()
    yield
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI + Quantum Research Platform",
        version="2.0.0",
        description="Autonomous LLM-orchestrated quantum and AI research backend",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_middleware(app)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
