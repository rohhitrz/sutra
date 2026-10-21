"""The /v1/health contract."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx2
import pytest
from httpx2 import ASGITransport, AsyncClient

from sutra.api.app import create_app


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_reports_ok_when_the_database_is_reachable(
    client: httpx2.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    async def ok() -> bool:
        return True

    monkeypatch.setattr("sutra.api.routes_health.ping", ok)
    response = await client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    components = {c["name"]: c for c in body["components"]}
    assert components["postgres"] == {"name": "postgres", "ok": True, "detail": None}
    # Tracing is reported so you can see whether traces are landing...
    assert components["tracing"]["ok"] is True


async def test_health_stays_ok_when_tracing_is_unavailable(
    client: httpx2.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    """...but it never fails the check.

    Losing traces degrades what you can see, not what you can serve. A health check that
    pulls the service out of a load balancer because the observability stack is down has
    inverted its own purpose.
    """

    async def ok() -> bool:
        return True

    class UnreachableTracer:
        def auth_check(self) -> bool:
            return False

    monkeypatch.setattr("sutra.api.routes_health.ping", ok)
    monkeypatch.setattr("sutra.api.routes_health.get_tracer", lambda: UnreachableTracer())
    response = await client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    tracing = next(c for c in body["components"] if c["name"] == "tracing")
    assert tracing["ok"] is True
    assert "dropped" in tracing["detail"]


async def test_health_returns_503_when_the_database_is_down(
    client: httpx2.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    async def down() -> bool:
        return False

    monkeypatch.setattr("sutra.api.routes_health.ping", down)
    response = await client.get("/v1/health")

    # A health check that says OK while Postgres is unreachable would keep a load balancer
    # sending traffic to a process that cannot serve it.
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"][0]["ok"] is False
    assert "make up" in body["components"][0]["detail"]


async def test_openapi_schema_is_served(client: httpx2.AsyncClient):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/health" in response.json()["paths"]
