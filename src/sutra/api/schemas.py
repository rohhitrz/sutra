"""Pydantic models for the HTTP boundary (CLAUDE.md: Pydantic models at every boundary)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthComponent(BaseModel):
    """Status of one dependency."""

    name: str
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """`GET /v1/health`.

    Reports per-dependency status rather than a bare 200. A health check that returns OK while
    the database is unreachable is worse than no health check — it tells a load balancer to keep
    sending traffic to a process that cannot serve it.
    """

    status: Literal["ok", "degraded"]
    version: str
    app_env: str
    components: list[HealthComponent] = Field(default_factory=list)
