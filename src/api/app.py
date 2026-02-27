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
from src.llm.master_llm import assert_master_llm_ready

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    logger.info("app.startup", env=settings.APP_ENV, version="2.0.0")
    await init_db()
    try:
        await assert_master_llm_ready()
        logger.info("app.startup.llm_ready", model=settings.huggingface_model_id)
    except Exception as exc:
        logger.critical("app.startup.llm_unavailable", error=str(exc))
        raise RuntimeError(f"Startup aborted: master LLM is not ready ({exc})") from exc
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
