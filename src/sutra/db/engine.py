"""Async engine and session factory.

All database I/O in Sutra is async (CLAUDE.md: no blocking calls inside request handlers).
`Settings` validates that DATABASE_URL uses the asyncpg driver, so a sync driver cannot be
configured by accident.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from sutra.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """One engine per process. The engine owns the connection pool, so creating more than one
    silently multiplies your connection count against Postgres."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.log_level == "DEBUG",
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        autoflush=False,
    )


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Commits on success, rolls back on any exception, always closes."""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ping() -> bool:
    """Cheapest possible liveness probe, used by /v1/health."""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def dispose_engine() -> None:
    """Close the pool on shutdown. Without this, uvicorn's reloader leaks connections."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
