from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from app.database import Database
from app.errors import DatabaseUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_NAME = "Nebula"
NAME_KEY = "assistant_name"
LOGO_KEY = "assistant_logo_file"
BACKGROUND_KEY = "assistant_background_enabled"
BACKGROUND_TYPE_KEY = "assistant_background_type"
BACKGROUND_VIDEO_KEY = "assistant_background_video"
BRANDING_KEYS = (
    NAME_KEY,
    LOGO_KEY,
    BACKGROUND_KEY,
    BACKGROUND_TYPE_KEY,
    BACKGROUND_VIDEO_KEY,
)


@dataclass(frozen=True)
class Branding:
    name: str
    logo_file: str | None
    background_enabled: bool
    background_type: str | None
    background_video_file: str | None
    updated_at: datetime | None


class BrandingStore:
    """Client-scoped assistant branding settings."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def _load(self, client_id: str) -> Branding:
        try:
            async with self.database.acquire() as connection:
                records = await connection.fetch(
                    """
                    SELECT setting_key, value, updated_at
                    FROM client_branding
                    WHERE client_id = $1
                      AND setting_key = ANY($2::varchar[])
                    """,
                    client_id,
                    list(BRANDING_KEYS),
                )
        except DatabaseUnavailableError:
            logger.warning(
                "Database unavailable; using default branding for client %s.",
                client_id,
            )
            records = []

        values = {record["setting_key"]: record for record in records}
        logo_file = str(values[LOGO_KEY]["value"]) if LOGO_KEY in values else None
        background_enabled = (
            str(values[BACKGROUND_KEY]["value"]).lower() == "true"
            if BACKGROUND_KEY in values
            else False
        )
        background_type = (
            str(values[BACKGROUND_TYPE_KEY]["value"]) or None
            if BACKGROUND_TYPE_KEY in values
            else None
        )
        background_video_file = (
            str(values[BACKGROUND_VIDEO_KEY]["value"])
            if BACKGROUND_VIDEO_KEY in values
            else None
        )
        if background_enabled and background_type is None:
            background_type = "video" if background_video_file else "image"
        timestamps = [record["updated_at"] for record in records]
        return Branding(
            name=(
                str(values[NAME_KEY]["value"])
                if NAME_KEY in values
                else DEFAULT_ASSISTANT_NAME
            ),
            logo_file=logo_file,
            background_enabled=background_enabled,
            background_type=background_type,
            background_video_file=background_video_file,
            updated_at=max(timestamps) if timestamps else None,
        )

    async def get(self, client_id: str) -> Branding:
        return await self._load(client_id)

    async def set_name(self, client_id: str, value: str) -> Branding:
        value = " ".join(value.split())
        if not value:
            raise ValueError("助手名称不能为空")
        if len(value) > 32:
            raise ValueError("助手名称不能超过 32 个字符")
        if any(ord(char) < 32 for char in value):
            raise ValueError("助手名称包含无效字符")
        await self._save_settings(client_id, ((NAME_KEY, value),))
        return await self._load(client_id)

    async def set_logo(self, client_id: str, asset_name: str) -> Branding:
        await self._save_settings(client_id, ((LOGO_KEY, asset_name),))
        return await self._load(client_id)

    async def set_background(
        self,
        client_id: str,
        enabled: bool,
        background_type: str | None = None,
    ) -> Branding:
        current = await self._load(client_id)
        if enabled:
            selected_type = background_type or (
                "video" if current.background_video_file else "image"
            )
            if selected_type == "image" and not current.logo_file:
                raise ValueError("请先上传助手图片")
            if selected_type == "video" and not current.background_video_file:
                raise ValueError("请先上传背景视频")
        else:
            selected_type = background_type or current.background_type
        await self._save_settings(
            client_id,
            (
                (BACKGROUND_KEY, "true" if enabled else "false"),
                (BACKGROUND_TYPE_KEY, selected_type or ""),
            ),
        )
        return await self._load(client_id)

    async def set_video(self, client_id: str, filename: str) -> Branding:
        await self._save_settings(
            client_id,
            (
                (BACKGROUND_VIDEO_KEY, filename),
                (BACKGROUND_TYPE_KEY, "video"),
                (BACKGROUND_KEY, "true"),
            ),
        )
        return await self._load(client_id)

    async def clear_video(self, client_id: str) -> Branding:
        current = await self._load(client_id)
        async with self.database.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    DELETE FROM client_branding
                    WHERE client_id = $1 AND setting_key = $2
                    """,
                    client_id,
                    BACKGROUND_VIDEO_KEY,
                )
                if current.background_type == "video":
                    await self._upsert(
                        connection,
                        client_id,
                        BACKGROUND_KEY,
                        "false",
                    )
        return await self._load(client_id)

    async def get_legacy_logo(
        self,
        asset_name: str,
    ) -> tuple[bytes, str] | None:
        async with self.database.acquire() as connection:
            record = await connection.fetchrow(
                "SELECT content_type, data FROM app_assets WHERE name = $1",
                asset_name,
            )
        if record is None:
            return None
        return bytes(record["data"]), str(record["content_type"])

    async def delete_legacy_logo(self, asset_name: str) -> None:
        async with self.database.acquire() as connection:
            await connection.execute(
                "DELETE FROM app_assets WHERE name = $1",
                asset_name,
            )

    async def clear_logo(self, client_id: str) -> Branding:
        current = await self._load(client_id)
        async with self.database.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    DELETE FROM client_branding
                    WHERE client_id = $1 AND setting_key = $2
                    """,
                    client_id,
                    LOGO_KEY,
                )
                if current.background_type == "image":
                    await self._upsert(
                        connection,
                        client_id,
                        BACKGROUND_KEY,
                        "false",
                    )
        return await self._load(client_id)

    async def _save_settings(
        self,
        client_id: str,
        entries: tuple[tuple[str, str], ...],
    ) -> None:
        async with self.database.acquire() as connection:
            async with connection.transaction():
                for key, value in entries:
                    await self._upsert(connection, client_id, key, value)

    @staticmethod
    async def _upsert(
        connection: asyncpg.Connection,
        client_id: str,
        key: str,
        value: str,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO client_branding (
                client_id, setting_key, value, updated_at
            )
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (client_id, setting_key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
            """,
            client_id,
            key,
            value,
        )
