from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings


@dataclass(frozen=True)
class StoredMedia:
    filename: str
    content_type: str
    size_bytes: int


def detect_image_type(content: bytes) -> tuple[str, str] | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    if (
        len(content) > 12
        and content.startswith(b"RIFF")
        and content[8:12] == b"WEBP"
    ):
        return ".webp", "image/webp"
    return None


def detect_video_type(
    content: bytes,
    declared_type: str | None,
) -> tuple[str, str] | None:
    if content.startswith(b"\x1aE\xdf\xa3"):
        return ".webm", "video/webm"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        if declared_type == "video/quicktime" or content[8:12] == b"qt  ":
            return ".mov", "video/quicktime"
        return ".mp4", "video/mp4"
    return None


class MediaStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def save_image(self, content: bytes) -> StoredMedia:
        detected = detect_image_type(content)
        if detected is None:
            raise ValueError("仅支持 PNG、JPEG、WebP 或 GIF 图片")
        extension, content_type = detected
        filename = f"assistant-{uuid4().hex}{extension}"
        self.settings.brand_media_dir.mkdir(parents=True, exist_ok=True)
        target = self.settings.brand_media_dir / filename
        await asyncio.to_thread(target.write_bytes, content)
        return StoredMedia(filename, content_type, len(content))

    async def save_legacy_image(
        self,
        filename: str,
        content: bytes,
    ) -> StoredMedia:
        detected = detect_image_type(content)
        if detected is None:
            raise ValueError("旧助手图片格式无效")
        _, content_type = detected
        suffix = Path(filename).suffix.lower()
        safe_name = f"assistant-{uuid4().hex}{suffix}"
        self.settings.brand_media_dir.mkdir(parents=True, exist_ok=True)
        target = self.settings.brand_media_dir / safe_name
        await asyncio.to_thread(target.write_bytes, content)
        return StoredMedia(safe_name, content_type, len(content))

    async def save_video(self, upload: UploadFile) -> StoredMedia:
        maximum = self.settings.max_background_video_mb * 1024 * 1024
        chunk_size = 1024 * 1024
        first_chunk = await upload.read(chunk_size)
        if not first_chunk:
            raise ValueError("视频文件为空")
        detected = detect_video_type(first_chunk, upload.content_type)
        if detected is None:
            raise ValueError("仅支持 MP4、MOV 或 WebM 视频")
        extension, content_type = detected
        filename = f"background-{uuid4().hex}{extension}"
        self.settings.background_media_dir.mkdir(parents=True, exist_ok=True)
        target = self.settings.background_media_dir / filename
        temp_path = target.with_suffix(f"{target.suffix}.part")
        handle = await asyncio.to_thread(temp_path.open, "wb")
        total = 0
        try:
            chunk = first_chunk
            while chunk:
                total += len(chunk)
                if total > maximum:
                    raise ValueError(
                        f"背景视频不能超过 {self.settings.max_background_video_mb} MB"
                    )
                await asyncio.to_thread(handle.write, chunk)
                chunk = await upload.read(chunk_size)
            await asyncio.to_thread(handle.flush)
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(temp_path.replace, target)
        except Exception:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(
                lambda: temp_path.unlink(missing_ok=True)
            )
            raise
        finally:
            await upload.close()
        return StoredMedia(filename, content_type, total)

    def media_path(self, directory: str, filename: str) -> Path | None:
        root = (
            self.settings.background_media_dir
            if directory == "backgrounds"
            else self.settings.brand_media_dir
        )
        target = (root / filename).resolve()
        if target.parent != root.resolve() or not target.is_file():
            return None
        return target

    async def delete(self, directory: str, filename: str | None) -> None:
        if not filename:
            return
        target = self.media_path(directory, filename)
        if target is not None:
            await asyncio.to_thread(target.unlink)

    def list_media(self, directory: str) -> list[dict[str, object]]:
        root = (
            self.settings.background_media_dir
            if directory == "backgrounds"
            else self.settings.brand_media_dir
        )
        if not root.is_dir():
            return []
        allowed = (
            {".mp4", ".mov", ".webm"}
            if directory == "backgrounds"
            else {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        )
        items: list[dict[str, object]] = []
        for path in root.iterdir():
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            stat = path.stat()
            items.append(
                {
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ),
                }
            )
        return sorted(
            items,
            key=lambda item: item["created_at"],
            reverse=True,
        )
