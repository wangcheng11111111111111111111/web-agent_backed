from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from app.config import Settings
from app.errors import DatabaseUnavailableError

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool: asyncpg.Pool | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self.pool is not None and not self.pool._closed:
            return
        async with self._connect_lock:
            if self.pool is not None and not self.pool._closed:
                return
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=self.settings.database_dsn,
                    min_size=1,
                    max_size=10,
                    command_timeout=20,
                )
            except Exception as exc:
                logger.warning(
                    "PostgreSQL connection unavailable (%s: %s). "
                    "Health checks remain available.",
                    type(exc).__name__,
                    str(exc),
                )
                raise DatabaseUnavailableError() from exc

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        if self.pool is None or self.pool._closed:
            try:
                await self.connect()
            except DatabaseUnavailableError:
                raise
        assert self.pool is not None
        try:
            async with self.pool.acquire() as connection:
                yield connection
        except (asyncpg.PostgresError, OSError) as exc:
            logger.error(
                "PostgreSQL operation failed (%s): %s",
                type(exc).__name__,
                str(exc),
            )
            raise DatabaseUnavailableError() from exc
