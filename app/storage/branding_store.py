from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.database import Database
from app.errors import DatabaseUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_NAME = "Nebula"
NAME_KEY = "assistant_name"
LOGO_KEY = "assistant_logo_file"
BACKGROUND_KEY = "assistant_background_enabled"
BACKGROUND_TYPE_KEY = "assistant_background_type"
BACKGROUND_VIDEO_KEY = "assistant_background_video"


@dataclass(frozen=True)
class Branding:
    name: str
    logo_file: str | None
    background_enabled: bool
    background_type: str | None
    background_video_file: str | None
    updated_at: datetime | None


class BrandingStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        self._name = DEFAULT_ASSISTANT_NAME
        self._logo_file: str | None = None
        self._background_enabled = False
        self._background_type: str | None = None
        self._background_video_file: str | None = None
        self._updated_at: datetime | None = None
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _load(self) -> None:
        try:
            async with self.database.acquire() as connection:
                records = await connection.fetch(
                    """
                    SELECT key, value, updated_at
                    FROM app_settings
                    WHERE key = ANY($1::varchar[])
                    """,
                    [
                        NAME_KEY,
                        LOGO_KEY,
                        BACKGROUND_KEY,
                        BACKGROUND_TYPE_KEY,
                        BACKGROUND_VIDEO_KEY,
                    ],
                )
        except DatabaseUnavailableError:
            logger.warning("Database unavailable; using default branding.")
            return
        values = {record["key"]: record for record in records}
        if NAME_KEY in values:
            self._name = str(values[NAME_KEY]["value"])
        if LOGO_KEY in values:
            self._logo_file = str(values[LOGO_KEY]["value"])
        if BACKGROUND_KEY in values:
            self._background_enabled = (
                str(values[BACKGROUND_KEY]["value"]).lower() == "true"
            )
        if BACKGROUND_TYPE_KEY in values:
            self._background_type = str(values[BACKGROUND_TYPE_KEY]["value"])
        if BACKGROUND_VIDEO_KEY in values:
            self._background_video_file = str(
                values[BACKGROUND_VIDEO_KEY]["value"]
            )
        timestamps = [
            record["updated_at"] for record in values.values()
        ]
        if timestamps:
            self._updated_at = max(timestamps)

    async def get(self) -> Branding:
        async with self._lock:
            if not self._loaded:
                await self._load()
                if self._background_enabled and self._background_type is None:
                    self._background_type = (
                        "video" if self._background_video_file else "image"
                    )
                self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

    async def set_name(self, value: str) -> Branding:
        value = " ".join(value.split())
        if not value:
            raise ValueError("助手名称不能为空")
        if len(value) > 32:
            raise ValueError("助手名称不能超过 32 个字符")
        if any(ord(char) < 32 for char in value):
            raise ValueError("助手名称包含无效字符")
        async with self._lock:
            await self._save_setting(NAME_KEY, value)
            self._name = value
            self._updated_at = datetime.now(timezone.utc)
            self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

    async def set_logo(
        self,
        asset_name: str,
    ) -> Branding:
        async with self._lock:
            if not self._loaded:
                await self._load()
            await self._save_setting(LOGO_KEY, asset_name)
            self._logo_file = asset_name
            self._updated_at = datetime.now(timezone.utc)
            self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

    async def set_background(
        self,
        enabled: bool,
        background_type: str | None = None,
    ) -> Branding:
        async with self._lock:
            if not self._loaded:
                await self._load()
            if enabled:
                selected_type = background_type or (
                    "video" if self._background_video_file else "image"
                )
                if selected_type == "image" and not self._logo_file:
                    raise ValueError("请先上传助手图片")
                if selected_type == "video" and not self._background_video_file:
                    raise ValueError("请先上传背景视频")
            else:
                selected_type = background_type or self._background_type
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    for key, value in (
                        (BACKGROUND_KEY, "true" if enabled else "false"),
                        (BACKGROUND_TYPE_KEY, selected_type or ""),
                    ):
                        await connection.execute(
                            """
                            INSERT INTO app_settings (key, value, updated_at)
                            VALUES ($1, $2, NOW())
                            ON CONFLICT (key) DO UPDATE SET
                                value = EXCLUDED.value,
                                updated_at = NOW()
                            """,
                            key,
                            value,
                        )
            self._background_enabled = enabled
            self._background_type = selected_type
            self._updated_at = datetime.now(timezone.utc)
            self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

    async def set_video(self, filename: str) -> Branding:
        async with self._lock:
            if not self._loaded:
                await self._load()
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    for key, value in (
                        (BACKGROUND_VIDEO_KEY, filename),
                        (BACKGROUND_TYPE_KEY, "video"),
                        (BACKGROUND_KEY, "true"),
                    ):
                        await connection.execute(
                            """
                            INSERT INTO app_settings (key, value, updated_at)
                            VALUES ($1, $2, NOW())
                            ON CONFLICT (key) DO UPDATE SET
                                value = EXCLUDED.value,
                                updated_at = NOW()
                            """,
                            key,
                            value,
                        )
            self._background_video_file = filename
            self._background_type = "video"
            self._background_enabled = True
            self._updated_at = datetime.now(timezone.utc)
            self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

    async def clear_video(self) -> Branding:
        async with self._lock:
            if not self._loaded:
                await self._load()
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(
                        "DELETE FROM app_settings WHERE key = $1",
                        BACKGROUND_VIDEO_KEY,
                    )
                    if self._background_type == "video":
                        await connection.execute(
                            """
                            INSERT INTO app_settings (key, value, updated_at)
                            VALUES ($1, 'false', NOW())
                            ON CONFLICT (key) DO UPDATE SET
                                value = EXCLUDED.value,
                                updated_at = NOW()
                            """,
                            BACKGROUND_KEY,
                        )
            self._background_video_file = None
            if self._background_type == "video":
                self._background_enabled = False
                self._background_type = None
            self._updated_at = datetime.now(timezone.utc)
            self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

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

    async def clear_logo(self) -> Branding:
        async with self._lock:
            if not self._loaded:
                await self._load()
            disable_image_background = self._background_type == "image"
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(
                        "DELETE FROM app_settings WHERE key = $1",
                        LOGO_KEY,
                    )
                    if disable_image_background:
                        await connection.execute(
                            """
                            INSERT INTO app_settings (key, value, updated_at)
                            VALUES ($1, 'false', NOW())
                            ON CONFLICT (key) DO UPDATE SET
                                value = EXCLUDED.value,
                                updated_at = NOW()
                            """,
                            BACKGROUND_KEY,
                        )
            self._logo_file = None
            if disable_image_background:
                self._background_enabled = False
                self._background_type = None
            self._updated_at = datetime.now(timezone.utc)
            self._loaded = True
            return Branding(
                self._name,
                self._logo_file,
                self._background_enabled,
                self._background_type,
                self._background_video_file,
                self._updated_at,
            )

    async def _save_setting(self, key: str, value: str) -> None:
        async with self.database.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
                """,
                key,
                value,
            )
