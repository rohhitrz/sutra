"""Health endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response, status

from sutra import __version__
from sutra.api.schemas import HealthComponent, HealthResponse
from sutra.config import get_settings
from sutra.db.engine import ping
from sutra.observability.tracing import get_tracer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    """Liveness plus dependency status.

    Returns 503 when a required dependency is down, so orchestrators and `make up` smoke checks
    can tell "process is listening" from "process can actually serve traffic".
    """
    settings = get_settings()
    db_ok = await ping()

    components = [
        HealthComponent(
            name="postgres",
            ok=db_ok,
            detail=None if db_ok else "SELECT 1 failed — is `make up` running?",
        )
    ]

    # Tracing is reported but never fails the check. Losing traces degrades what you can see,
    # not what you can serve, and a health check that takes the service out of a load
    # balancer because the observability stack is down has inverted its own purpose.
    tracer = get_tracer()
    auth_check = getattr(tracer, "auth_check", None)
    if auth_check is None:
        tracing_detail = "off" if not settings.tracing_enabled else "no Langfuse keys configured"
        components.append(HealthComponent(name="tracing", ok=True, detail=tracing_detail))
    else:
        reachable = await asyncio.to_thread(auth_check)
        components.append(
            HealthComponent(
                name="tracing",
                ok=True,
                detail=(
                    f"langfuse at {settings.langfuse_host}"
                    if reachable
                    else f"langfuse at {settings.langfuse_host} unreachable — "
                    "traces are being dropped"
                ),
            )
        )

    # `required` rather than every component, for the reason above.
    required = [c for c in components if c.name != "tracing"]
    overall_ok = all(component.ok for component in required)
    if not overall_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        version=__version__,
        app_env=settings.app_env,
        components=components,
    )
