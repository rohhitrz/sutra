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
    assert body["components"] == [{"name": "postgres", "ok": True, "detail": None}]


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
