from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import UUID

from redis.asyncio import Redis

from app.config import Settings

logger = logging.getLogger(__name__)

APPEND_SCRIPT = """
local key = KEYS[1]
local exists = redis.call('EXISTS', key)
if exists == 0 and ARGV[2] == '0' then
    return 0
end
redis.call('RPUSH', key, ARGV[1])
redis.call('LTRIM', key, -tonumber(ARGV[3]), -1)
redis.call('EXPIRE', key, tonumber(ARGV[4]))
return 1
"""


class ShortTermMemory:
    """Best-effort Redis cache for recent session messages.

    PostgreSQL remains the source of truth. Redis failures are logged and
    degrade to database reads/writes without failing chat requests.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Redis | None = None
        self._retry_after = 0.0

    @property
    def enabled(self) -> bool:
        return self.settings.redis_enabled

    async def connect(self) -> bool:
        if not self.enabled:
            return False
        if self.client is not None:
            return True
        if time.monotonic() < self._retry_after:
            return False
        try:
            client = Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await client.ping()
            self.client = client
            logger.info("Redis short-term memory connected")
            return True
        except Exception as exc:
            self._retry_after = time.monotonic() + 30
            logger.warning(
                "Redis unavailable; database fallback is active (%s)",
                type(exc).__name__,
            )
            return False

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    @staticmethod
    def _messages_key(session_id: UUID) -> str:
        return f"session:{session_id}:messages"

    @staticmethod
    def _meta_key(session_id: UUID) -> str:
        return f"session:{session_id}:meta"

    async def get_history(
        self, session_id: UUID
    ) -> list[dict[str, Any]] | None:
        if not await self.connect():
            return None
        assert self.client is not None
        try:
            key = self._messages_key(session_id)
            key_type = await self.client.type(key)
            if key_type == "none":
                return None
            if key_type == "list":
                raw_messages = await self.client.lrange(key, 0, -1)
                return [json.loads(raw) for raw in raw_messages]
            if key_type == "string":
                # Transparently migrate the previous whole-JSON cache format.
                raw = await self.client.get(key)
                value = json.loads(raw) if raw else []
                if not isinstance(value, list):
                    return None
                await self.cache_history(session_id, value)
                return value
            logger.warning("Unexpected Redis cache type %s; rebuilding", key_type)
            await self.client.delete(key)
            return None
        except Exception as exc:
            self._mark_failure(exc)
            return None

    async def cache_history(
        self,
        session_id: UUID,
        messages: list[dict[str, Any]],
        *,
        title: str | None = None,
    ) -> None:
        if not await self.connect():
            return
        assert self.client is not None
        limit = self.settings.short_term_message_limit
        ttl = self.settings.session_ttl_seconds
        try:
            pipe = self.client.pipeline(transaction=True)
            pipe.delete(self._messages_key(session_id))
            serialized = [
                json.dumps(message, ensure_ascii=False)
                for message in messages[-limit:]
            ]
            if serialized:
                pipe.rpush(self._messages_key(session_id), *serialized)
                pipe.expire(self._messages_key(session_id), ttl)
            if title is not None:
                pipe.hset(
                    self._meta_key(session_id),
                    mapping={"title": title, "updated_at": time.time()},
                )
                pipe.expire(self._meta_key(session_id), ttl)
            await pipe.execute()
        except Exception as exc:
            self._mark_failure(exc)

    async def append_message(
        self,
        session_id: UUID,
        message: dict[str, Any],
        *,
        create_if_missing: bool = False,
    ) -> bool:
        if not await self.connect():
            return False
        assert self.client is not None
        try:
            result = await self.client.eval(
                APPEND_SCRIPT,
                1,
                self._messages_key(session_id),
                json.dumps(message, ensure_ascii=False),
                "1" if create_if_missing else "0",
                self.settings.short_term_message_limit,
                self.settings.session_ttl_seconds,
            )
            return bool(result)
        except Exception as exc:
            self._mark_failure(exc)
            return False

    async def delete_session(self, session_id: UUID) -> None:
        if not await self.connect():
            return
        assert self.client is not None
        try:
            await self.client.delete(
                self._messages_key(session_id),
                self._meta_key(session_id),
            )
        except Exception as exc:
            self._mark_failure(exc)

    def _mark_failure(self, exc: Exception) -> None:
        logger.warning(
            "Redis operation failed; continuing with PostgreSQL (%s)",
            type(exc).__name__,
        )
        self._retry_after = time.monotonic() + 30
