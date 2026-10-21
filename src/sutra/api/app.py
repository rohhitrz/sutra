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
from sutra.api.routes_chat import router as chat_router
from sutra.api.routes_feedback import router as feedback_router
from sutra.api.routes_health import router as health_router
from sutra.config import get_settings
from sutra.db.engine import dispose_engine
from sutra.observability.logs import configure_logging
from sutra.observability.tracing import build_tracer, get_tracer, set_tracer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown.

    The engine is created lazily on first use, so startup does not require a reachable
    database — `/v1/health` is then able to *report* that the database is down, which is more
    useful than refusing to boot.
    """
    settings = get_settings()
    configure_logging(settings.log_level, log_format=settings.log_format)
    # The process-wide tracer is built once, here, and read through `get_tracer()` by code at
    # any depth (see the CONCEPT note in `observability.tracing`). It is a working no-op when
    # tracing is off or unconfigured, so nothing downstream branches on whether it is real.
    set_tracer(build_tracer(settings))
    logger.info(
        "sutra.startup",
        extra={
            "version": __version__,
            "env": settings.app_env,
            "tracing": settings.tracing_enabled,
        },
    )
    try:
        yield
    finally:
        # Flush before the engine goes: the exporter batches, so spans from the last request
        # in flight are still buffered and would otherwise be dropped on exit.
        get_tracer().shutdown()
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
    app.include_router(chat_router, prefix="/v1")
    app.include_router(feedback_router, prefix="/v1")
    return app


app = create_app()
