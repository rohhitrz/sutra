"""FastAPI application factory.

Every client — HTTP, voice, MCP — eventually goes through the same core (ARCHITECTURE §1), so
this module stays thin: wiring, lifespan and routers only. No business logic.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sutra import __version__
from sutra.api.routes_health import router as health_router
from sutra.config import get_settings
from sutra.db.engine import dispose_engine

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Plain logging for now. Phase 4 replaces this with structured JSON logs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown.

    The engine is created lazily on first use, so startup does not require a reachable
    database — `/v1/health` is then able to *report* that the database is down, which is more
    useful than refusing to boot.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("sutra.startup version=%s env=%s", __version__, settings.app_env)
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("sutra.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sutra",
        version=__version__,
        summary="Question answering over a fictional story corpus, with citations.",
        lifespan=lifespan,
    )
    app.include_router(health_router, prefix="/v1")
    return app


app = create_app()
