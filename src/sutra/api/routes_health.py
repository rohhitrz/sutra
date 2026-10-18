"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from sutra import __version__
from sutra.api.schemas import HealthComponent, HealthResponse
from sutra.config import get_settings
from sutra.db.engine import ping

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
    overall_ok = all(component.ok for component in components)
    if not overall_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        version=__version__,
        app_env=settings.app_env,
        components=components,
    )
