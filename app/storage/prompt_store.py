from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.config import DEFAULT_PERSONALITY_PROMPT, SYSTEM_RULES_PROMPT
from app.errors import DatabaseUnavailableError

if TYPE_CHECKING:
    from app.database import Database

logger = logging.getLogger(__name__)

PERSONALITY_KEY = "personality_prompt"


class PromptStore:
    def __init__(self, database: Database | None = None) -> None:
        self._database = database
        self._personality: str | None = None
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _load_from_db(self) -> str | None:
        if self._database is None:
            return None
        try:
            async with self._database.acquire() as connection:
                record = await connection.fetchrow(
                    "SELECT value FROM app_settings WHERE key = $1",
                    PERSONALITY_KEY,
                )
        except DatabaseUnavailableError:
            logger.warning("Database unavailable; using in-memory personality prompt.")
            return None
        if record is None:
            return None
        return str(record["value"])

    async def _persist_to_db(self, value: str | None) -> bool:
        if self._database is None:
            return False
        try:
            async with self._database.acquire() as connection:
                if value is None:
                    await connection.execute(
                        "DELETE FROM app_settings WHERE key = $1",
                        PERSONALITY_KEY,
                    )
                else:
                    await connection.execute(
                        """
                        INSERT INTO app_settings (key, value, updated_at)
                        VALUES ($1, $2, NOW())
                        ON CONFLICT (key) DO UPDATE SET
                            value = EXCLUDED.value,
                            updated_at = NOW()
                        """,
                        PERSONALITY_KEY,
                        value,
                    )
            return True
        except DatabaseUnavailableError:
            logger.warning(
                "Database unavailable; personality prompt not persisted."
            )
            return False

    async def get_personality(self) -> str:
        async with self._lock:
            if not self._loaded:
                stored = await self._load_from_db()
                if stored is not None:
                    self._personality = stored
                self._loaded = True
            return (
                self._personality
                if self._personality is not None
                else DEFAULT_PERSONALITY_PROMPT
            )

    async def get_full_system_prompt(self) -> str:
        personality = await self.get_personality()
        return personality + SYSTEM_RULES_PROMPT

    async def set_personality(self, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("性格提示词不能为空")
        async with self._lock:
            self._personality = value
            self._loaded = True
        await self._persist_to_db(value)
        logger.info("Personality prompt updated (length=%s)", len(value))
        return value

    async def reset_personality(self) -> str:
        async with self._lock:
            self._personality = None
            self._loaded = True
        await self._persist_to_db(None)
        logger.info("Personality prompt reset to default")
        return DEFAULT_PERSONALITY_PROMPT

    @property
    def is_custom(self) -> bool:
        return self._personality is not None


_store: PromptStore | None = None


def get_prompt_store() -> PromptStore:
    global _store
    if _store is None:
        _store = PromptStore()
    return _store