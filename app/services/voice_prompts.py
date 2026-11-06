from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

MAX_PROMPT_BYTES = 20 * 1024 * 1024
MIN_PROMPT_DURATION_SECONDS = 1.0
MAX_PROMPT_DURATION_SECONDS = 30.0
ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _normalize_reference_text(text: str) -> str:
    """Keep prompt text aligned with speech and remove unspeakable noise."""
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    text = re.sub(r"```[\s\S]*?```", lambda match: match.group(0).replace("```", ""), text)
    text = text.replace("`", "")
    text = re.sub(r"[*#|>]+", "", text)
    text = re.sub(
        r"(?<=\d)\s*(?:-{2,}|[—–−]+)\s*(?=\d)",
        "到",
        text,
    )
    text = re.sub(r"\s*(?:-{2,}|[—–−]+)\s*", "，", text)
    text = text.replace("、", "，")
    text = re.sub(
        r"[^\u3400-\u9fffA-Za-z0-9，。！？；：,.!?;:'\"“”‘’（）()《》〈〉…\s]",
        "",
        text,
    )
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n+\s*", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"，([。！？!?；;])", r"\1", text)
    text = re.sub(r"[ \t]+([，。！？!?；;：:])", r"\1", text)
    return text.strip()


class VoicePromptStore:
    def __init__(
        self,
        settings: Settings,
        *,
        default_wav: Path,
        default_text: str,
    ) -> None:
        self.settings = settings
        self.root = settings.brand_media_dir.parent / "voice_prompts"
        self.metadata_path = self.root / "prompts.json"
        self.default_wav = default_wav
        self.default_text = default_text
        self._lock = threading.Lock()
        self._whisper_lock = threading.Lock()
        self._whisper_model: Any = None
        self._whisper_model_key: tuple[str, str, bool] | None = None

    def _records(self) -> list[dict[str, Any]]:
        if not self.metadata_path.is_file():
            return []
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            logger.warning("Ignoring invalid voice prompt metadata", exc_info=True)
            return []

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.metadata_path)

    def list_prompts(self) -> list[dict[str, Any]]:
        default = {
            "id": "default",
            "name": "默认语气",
            "text": self.default_text,
            "text_source": "builtin",
            "duration": None,
            "created_at": None,
            "builtin": True,
            "audio_url": "/api/audio/prompts/default/audio",
        }
        with self._lock:
            records = self._records()
        custom = [
            {
                **record,
                "text": _normalize_reference_text(
                    str(record.get("text", ""))
                ),
                "builtin": False,
                "audio_url": f"/api/audio/prompts/{record['id']}/audio",
            }
            for record in sorted(
                records,
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
        ]
        return [default, *custom]

    def get(self, prompt_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next(
                (
                    record
                    for record in self._records()
                    if record.get("id") == prompt_id
                ),
                None,
            )

    def resolve(
        self,
        prompt_id: str | None,
        *,
        require_text: bool = True,
    ) -> tuple[Path, str]:
        if not prompt_id or prompt_id == "default":
            return self.default_wav, self.default_text
        record = self.get(prompt_id)
        if record is None:
            raise ValueError("语气预设不存在")
        self._validate_prompt_duration(record.get("duration"))
        audio_path = self.root / str(record["filename"])
        if not audio_path.is_file():
            raise ValueError("语气预设音频文件不存在")
        text = _normalize_reference_text(
            str(record.get("text", ""))
        )
        if require_text and not text:
            raise ValueError("该语气预设还没有参考文本，请先填写或重新识别")
        duration = record.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            maximum = max(30, math.ceil(float(duration) * 8))
            if len(text) > maximum:
                raise ValueError(
                    "参考文本与音频时长不匹配，"
                    f"最长建议 {maximum} 字，当前 {len(text)} 字"
                )
        return audio_path, text

    def audio_path(self, prompt_id: str) -> Path:
        if prompt_id == "default":
            if not self.default_wav.is_file():
                raise ValueError("内置语气音频文件不存在")
            return self.default_wav
        record = self.get(prompt_id)
        if record is None:
            raise ValueError("语气预设不存在")
        path = self.root / str(record["filename"])
        if not path.is_file():
            raise ValueError("语气预设音频文件不存在")
        return path

    async def add(
        self,
        *,
        filename: str,
        content: bytes,
        name: str = "",
        text: str = "",
        duration_limit: float = 15.0,
    ) -> tuple[dict[str, Any], str | None]:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError("语气参考音频仅支持 WAV、MP3、FLAC、OGG、M4A")
        if not content:
            raise ValueError("上传音频不能为空")
        if len(content) > MAX_PROMPT_BYTES:
            raise ValueError("语气参考音频不能超过 20 MB")

        prompt_id = uuid.uuid4().hex
        stored_name = f"{prompt_id}{suffix}"
        audio_path = self.root / stored_name
        self.root.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(content)
        try:
            duration = await asyncio.to_thread(self._audio_duration, audio_path)
        except Exception as exc:
            try:
                audio_path.unlink()
            except OSError:
                logger.warning("Invalid prompt audio cleanup failed", exc_info=True)
            raise ValueError("无法读取音频，请确认文件未损坏且格式受支持") from exc
        if duration < MIN_PROMPT_DURATION_SECONDS:
            try:
                audio_path.unlink()
            except OSError:
                logger.warning("Invalid prompt audio cleanup failed", exc_info=True)
            raise ValueError("参考音频太短，请使用至少 1 秒的清晰语音")
        duration_limit = max(
            MIN_PROMPT_DURATION_SECONDS,
            min(float(duration_limit), MAX_PROMPT_DURATION_SECONDS),
        )
        if duration > duration_limit:
            try:
                audio_path.unlink()
            except OSError:
                logger.warning("Invalid prompt audio cleanup failed", exc_info=True)
            raise ValueError(
                f"参考音频时长为 {duration:.1f} 秒，"
                f"超过所选 {duration_limit:g} 秒上限，请先裁剪"
            )

        recognition_warning: str | None = None
        text = _normalize_reference_text(text)
        text_source = "manual"
        if not text:
            try:
                text = _normalize_reference_text(
                    await self._recognize(audio_path, filename)
                )
                text_source = "recognized"
            except Exception as exc:
                logger.warning("Voice prompt recognition failed: %s", exc)
                recognition_warning = (
                    f"自动识别失败：{exc}。请手动填写参考文本。"
                )
                text_source = "missing"

        record = {
            "id": prompt_id,
            "name": (name.strip() or Path(filename).stem or "未命名语气")[:64],
            "filename": stored_name,
            "original_filename": Path(filename).name,
            "text": text,
            "text_source": text_source,
            "duration": duration,
            "duration_limit": duration_limit,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            records = self._records()
            records.append(record)
            self._write_records(records)
        return {
            **record,
            "builtin": False,
            "audio_url": f"/api/audio/prompts/{prompt_id}/audio",
        }, recognition_warning

    async def transcribe(self, prompt_id: str) -> dict[str, Any]:
        record = self.get(prompt_id)
        if record is None:
            raise ValueError("语气预设不存在")
        self._validate_prompt_duration(record.get("duration"))
        path = self.audio_path(prompt_id)
        text = await self._recognize(path, str(record.get("original_filename", "")))
        return self.update(prompt_id, text=text, text_source="recognized")

    def update(
        self,
        prompt_id: str,
        *,
        name: str | None = None,
        text: str | None = None,
        text_source: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = self._records()
            record = next(
                (item for item in records if item.get("id") == prompt_id),
                None,
            )
            if record is None:
                raise ValueError("语气预设不存在")
            if name is not None and name.strip():
                record["name"] = name.strip()[:64]
            if text is not None:
                record["text"] = _normalize_reference_text(text)
                record["text_source"] = (
                    text_source
                    if text_source is not None
                    else ("manual" if record["text"] else "missing")
                )
            self._write_records(records)
        return {
            **record,
            "builtin": False,
            "audio_url": f"/api/audio/prompts/{prompt_id}/audio",
        }

    def delete(self, prompt_id: str) -> None:
        with self._lock:
            records = self._records()
            record = next(
                (item for item in records if item.get("id") == prompt_id),
                None,
            )
            if record is None:
                raise ValueError("语气预设不存在")
            records = [item for item in records if item.get("id") != prompt_id]
            self._write_records(records)
        filenames = {
            str(record.get("filename", "")),
            str(record.get("source_filename", "")),
            str(record.get("prompt_filename", "")),
        }
        for filename in filenames:
            if not filename:
                continue
            try:
                (self.root / filename).unlink()
            except OSError:
                logger.warning("Voice prompt audio deletion failed", exc_info=True)

    @staticmethod
    def _audio_duration(path: Path) -> float:
        import soundfile as sf

        try:
            info = sf.info(str(path))
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            # libsndfile does not decode every browser-supported container.
            import librosa

            duration = float(librosa.get_duration(path=str(path)))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("音频中不包含有效声音")
        return round(duration, 3)

    @staticmethod
    def _validate_prompt_duration(value: object) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return
        duration = float(value)
        if duration < MIN_PROMPT_DURATION_SECONDS:
            raise ValueError("参考音频太短，请使用至少 1 秒的清晰语音")
        if duration > MAX_PROMPT_DURATION_SECONDS:
            raise ValueError(
                "参考音频超过 30 秒。CosyVoice2 零样本提示建议使用 3–15 秒，"
                f"当前为 {duration:.1f} 秒，请先裁剪后再上传"
            )

    async def _recognize(self, path: Path, filename: str) -> str:
        if self.settings.stt_enabled and self.settings.stt_api_key:
            try:
                return await self._recognize_remote(path, filename)
            except Exception:
                logger.warning("Remote STT failed; trying local Whisper", exc_info=True)
        return await asyncio.to_thread(self._recognize_local, path)

    async def _recognize_remote(self, path: Path, filename: str) -> str:
        url = f"{self.settings.stt_base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.settings.stt_api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.stt_timeout) as client:
            response = await client.post(
                url,
                headers=headers,
                data={"model": self.settings.stt_model},
                files={
                    "file": (
                        Path(filename).name or path.name,
                        path.read_bytes(),
                        "application/octet-stream",
                    )
                },
            )
            response.raise_for_status()
            payload = response.json()
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("STT 未返回参考文本")
        return text.strip()

    def _recognize_local(self, path: Path) -> str:
        import librosa
        import torch
        import whisper

        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        device, use_fp16 = self._resolve_whisper_device()
        requested_model = self.settings.stt_whisper_model or "small"

        with self._whisper_lock:
            cached_model = (
                self._whisper_model_key[0]
                if self._whisper_model_key is not None
                else None
            )
            if self._whisper_model is None or cached_model != requested_model:
                model_name = requested_model
                try:
                    self._whisper_model = whisper.load_model(
                        model_name,
                        device=device,
                        download_root=str(
                            self.settings.brand_media_dir.parent / "whisper"
                        ),
                    )
                except Exception:
                    if device == "cpu":
                        raise
                    logger.warning(
                        "Whisper could not run on %s; falling back to CPU",
                        device,
                        exc_info=True,
                    )
                    torch.cuda.empty_cache()
                    device = "cpu"
                    use_fp16 = False
                    model_name = requested_model
                    self._whisper_model = whisper.load_model(
                        model_name,
                        device=device,
                        download_root=str(
                            self.settings.brand_media_dir.parent / "whisper"
                        ),
                    )
                self._whisper_model_key = (model_name, device, use_fp16)
                logger.info(
                    "Local Whisper using %s on %s (fp16=%s)",
                    model_name,
                    device,
                    use_fp16,
                )

            if self._whisper_model_key is None:
                raise RuntimeError("Whisper 模型初始化失败")
            _, device, use_fp16 = self._whisper_model_key
            result = self._whisper_model.transcribe(
                audio,
                language="zh",
                task="transcribe",
                fp16=use_fp16,
                condition_on_previous_text=False,
                without_timestamps=True,
                verbose=False,
            )
        text = str(result.get("text", "")).strip()
        if not text:
            raise RuntimeError("Whisper 未返回参考文本")
        return text

    def _resolve_whisper_device(self) -> tuple[str, bool]:
        import torch

        preference = self.settings.stt_device
        if preference != "cpu" and torch.cuda.is_available():
            if preference == "auto":
                return "cuda", True
            if preference.startswith("cuda"):
                return preference, True
        return "cpu", False
