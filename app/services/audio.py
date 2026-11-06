from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator

from app.config import BACKEND_DIR, Settings
from app.errors import ModelUpstreamError
from app.services.voice_prompts import VoicePromptStore

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 800
_EDGE_VOICE = "zh-CN-XiaoxiaoNeural"

EDGE_CN_VOICES: dict[str, str] = {
    "zh-CN-XiaoxiaoNeural": "晓晓（女声，亲和）",
    "zh-CN-XiaoyiNeural": "晓伊（女声，柔和）",
    "zh-CN-YunxiNeural": "云希（男声，标准）",
    "zh-CN-YunjianNeural": "云健（男声，阳光）",
    "zh-CN-YunxiaNeural": "云夏（男声，磁性）",
    "zh-CN-YunyangNeural": "云扬（男声，播报）",
    "zh-CN-liaoning-XiaobeiNeural": "小北（东北口音）",
    "zh-CN-shaanxi-XiaoniNeural": "小妮（陕西口音）",
    "zh-TW-HsiaoChenNeural": "晓辰（台湾女声）",
    "zh-TW-HsiaoYuNeural": "晓雨（台湾女声）",
    "zh-TW-YunJheNeural": "云哲（台湾男声）",
    "zh-HK-HiuGaaiNeural": "曉佳（香港粤语女声）",
    "zh-HK-HiuMaanNeural": "曉曼（香港粤语女声）",
    "zh-HK-WanLungNeural": "雲龍（香港粤语男声）",
}

REQUIRED_MODEL_FILES = (
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
)

COSYVOICE_VENDOR_DIR = BACKEND_DIR / "vendor" / "CosyVoice"
COSYVOICE_MATCHA_DIR = COSYVOICE_VENDOR_DIR / "third_party" / "Matcha-TTS"
RVC_VENDOR_DIR = BACKEND_DIR / "vendor" / "RVC"
DEFAULT_PROMPT_WAV = str(COSYVOICE_VENDOR_DIR / "asset" / "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

# The offline RVC WebUI disables CUDA Graph by default. Graph capture adds
# significant latency for variable-length one-shot audio and can consume the
# remaining VRAM when CosyVoice2 is already resident.
os.environ.setdefault("RVC_CUDA_GRAPH", "0")


def _ensure_vendor_on_path() -> None:
    paths = (COSYVOICE_VENDOR_DIR, COSYVOICE_MATCHA_DIR, RVC_VENDOR_DIR)
    for p in paths:
        s = str(p)
        if p.exists():
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)

    vendor_root = COSYVOICE_VENDOR_DIR.resolve()
    package = sys.modules.get("cosyvoice")
    module_file = getattr(package, "__file__", None)
    if package is not None and (
        not module_file
        or not Path(module_file).resolve().is_relative_to(vendor_root)
    ):
        for module_name in tuple(sys.modules):
            if module_name == "cosyvoice" or module_name.startswith("cosyvoice."):
                sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()


def _get_device(preference: str) -> str:
    try:
        import torch
        if preference == "cpu" or not torch.cuda.is_available():
            return "cpu"
        return "cuda"
    except ImportError:
        return "cpu"


def _scan_rvc_models(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.pth"))


def _split_tts_text(
    text: str,
    max_chars: int = 90,
    first_max_chars: int = 32,
) -> list[str]:
    """Split long replies into sentence-aligned chunks for progressive audio."""
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])|(?<=\n)", text)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for part in parts:
        remaining = part
        while remaining:
            limit = first_max_chars if not chunks else max_chars
            if not current and len(remaining) <= limit:
                current = remaining
                remaining = ""
            elif current and len(current) + len(remaining) <= limit:
                current += remaining
                remaining = ""
            elif current:
                chunks.append(current)
                current = ""
            else:
                chunks.append(remaining[:limit].strip())
                remaining = remaining[limit:]
    if current:
        chunks.append(current)
    return chunks or [text]


def _take_tts_chunk(text: str, limit: int) -> tuple[str, str]:
    """Take one chunk, preferring a nearby sentence or clause boundary."""
    if len(text) <= limit:
        return text.strip(), ""
    window = text[: limit + 1]
    minimum = max(12, limit // 3)
    for separators in ("。！？!?；;", "，,、：:"):
        positions = [window.rfind(char) for char in separators]
        cut = max(positions)
        if cut >= minimum:
            return text[: cut + 1].strip(), text[cut + 1 :].strip()
    return text[:limit].strip(), text[limit:].strip()


def _split_tts_sentences(
    text: str,
    max_sentence_chars: int = 72,
) -> list[str]:
    """Split by complete sentences and split long sentences into at most two."""
    boundaries = list(re.finditer(r"[。！？!?；;]+|\n+", text))
    raw_sentences: list[str] = []
    start = 0
    for match in boundaries:
        sentence = text[start : match.end()].strip()
        if sentence:
            raw_sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        raw_sentences.append(tail)

    segments: list[str] = []
    for sentence in raw_sentences or [text.strip()]:
        if len(sentence) <= max_sentence_chars:
            segments.append(sentence)
            continue
        midpoint = len(sentence) // 2
        search_start = max(8, midpoint - 20)
        search_end = min(len(sentence) - 8, midpoint + 20)
        candidates = [
            index
            for index, char in enumerate(sentence)
            if search_start <= index < search_end and char in "，,、：:"
        ]
        cut = (
            min(candidates, key=lambda index: abs(index - midpoint))
            if candidates
            else midpoint
        )
        first = sentence[: cut + 1].strip()
        second = sentence[cut + 1 :].strip()
        if first:
            segments.append(first)
        if second:
            segments.append(second)
    return segments or [text.strip()]


def _group_sentence_segments(
    segments: list[str],
    maximum_chars: int = 84,
) -> list[str]:
    """Pack complete sentences into coherent blocks."""
    grouped: list[str] = []
    current = ""
    for segment in segments:
        if not current:
            current = segment
            continue
        can_merge = len(current) + len(segment) <= maximum_chars
        if can_merge:
            current += segment
        else:
            grouped.append(current)
            current = segment
    if current:
        grouped.append(current)
    return grouped


def _normalize_spoken_text(text: str) -> str:
    """Convert markdown and decorative punctuation into speakable pauses."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Markdown table separator, horizontal rule, and code fences.
    text = re.sub(
        r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
        "\n",
        text,
    )
    text = re.sub(r"(?m)^\s*[-*_]{2,}\s*$", "\n", text)
    text = text.replace("```", "")
    text = text.replace("`", "")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"[*#]+", "", text)
    text = text.replace("|", "，")

    # Decorative dashes and Chinese list commas are pauses, not words.
    text = re.sub(
        r"(?<=\d)\s*(?:-{2,}|[—–−]+)\s*(?=\d)",
        "到",
        text,
    )
    text = re.sub(r"\s*(?:-{2,}|[—–−]+)\s*", "，", text)
    text = text.replace("、", "，")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"，([。！？!?；;])", r"\1", text)
    text = re.sub(r"[ \t]+([，。！？!?；;])", r"\1", text)
    return text.strip()


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._warmed_up = False
        self._warming_up = False
        self._engine_type = (
            settings.local_tts_engine
            if settings.local_tts_engine in {"edge_tts", "cosyvoice"}
            else "edge_tts"
        )
        self._cosyvoice_model: object | None = None
        self._cosyvoice_loading = threading.Lock()
        self._cosyvoice_import_error: str | None = None
        self._last_cosyvoice_import_error: str | None = None
        self._prompt_cache_keys: dict[str, tuple[str, int, int, str]] = {}
        self._rvc_engine: _RVCEngine | None = None
        self._synthesize_lock = asyncio.Lock()
        self._rvc_disabled = False
        self.prompts = VoicePromptStore(
            settings,
            default_wav=Path(DEFAULT_PROMPT_WAV),
            default_text=DEFAULT_PROMPT_TEXT,
        )

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> Path:
        return _resolve_path(self.settings.local_tts_model_dir)

    @property
    def rvc_model_dir(self) -> Path:
        return _resolve_path(self.settings.local_rvc_model_dir)

    @property
    def rvc_index_dir(self) -> Path:
        return _resolve_path(self.settings.local_rvc_index_dir)

    # ------------------------------------------------------------------
    # CosyVoice2 检测
    # ------------------------------------------------------------------

    def _check_cosyvoice_importable(self) -> bool:
        _ensure_vendor_on_path()
        try:
            import torch  # noqa: F401
            from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: F401
            self._cosyvoice_import_error = None
            return True
        except Exception as exc:
            self._cosyvoice_import_error = f"{type(exc).__name__}: {exc}"
            if self._cosyvoice_import_error != self._last_cosyvoice_import_error:
                logger.warning(
                    "CosyVoice2 import check failed: %s",
                    self._cosyvoice_import_error,
                )
                self._last_cosyvoice_import_error = self._cosyvoice_import_error
            return False

    def _check_cosyvoice_downloaded(self) -> bool:
        model_dir = self.model_dir
        if not model_dir.exists():
            return False
        for name in REQUIRED_MODEL_FILES:
            if not (model_dir / name).is_file():
                return False
        return True

    @property
    def missing_model_files(self) -> list[str]:
        return [
            name
            for name in REQUIRED_MODEL_FILES
            if not (self.model_dir / name).is_file()
        ]

    # ------------------------------------------------------------------
    # 引擎可用性
    # ------------------------------------------------------------------

    @staticmethod
    def _check_edge_tts_available() -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def configured(self) -> bool:
        if not self.settings.tts_enabled:
            return False
        return self._check_edge_tts_available() or (
            self._check_cosyvoice_importable()
            and self._check_cosyvoice_downloaded()
        )

    @property
    def rvc_enabled(self) -> bool:
        return not self._rvc_disabled and bool(_scan_rvc_models(self.rvc_model_dir))

    def set_rvc_enabled(self, enabled: bool) -> bool:
        self._rvc_disabled = not enabled
        if not enabled:
            self._rvc_engine = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return self.rvc_enabled

    # ------------------------------------------------------------------
    # 引擎列表（供前端选择）
    # ------------------------------------------------------------------

    @property
    def available_engines(self) -> list[dict]:
        engines: list[dict] = []

        cos_importable = self._check_cosyvoice_importable()
        cos_downloaded = cos_importable and self._check_cosyvoice_downloaded()

        engines.append({
            "id": "cosyvoice",
            "name": "CosyVoice2",
            "description": "高质量本地 TTS（需下载模型 ~1GB，需要 GPU）",
            "available": cos_importable,
            "downloaded": cos_downloaded,
            "needs_download": cos_importable and not cos_downloaded,
            "missing_files": self.missing_model_files if cos_importable and not cos_downloaded else [],
            "error": self._cosyvoice_import_error,
        })

        edge_ok = self._check_edge_tts_available()
        engines.append({
            "id": "edge_tts",
            "name": "Edge TTS",
            "description": "微软 Edge 免费语音（零显存，即开即用）",
            "available": edge_ok,
            "downloaded": True,
            "needs_download": False,
            "missing_files": [],
        })

        return engines

    @property
    def active_engine(self) -> str:
        return self._engine_type

    # ------------------------------------------------------------------
    # 音色列表
    # ------------------------------------------------------------------

    @property
    def available_voices(self) -> dict[str, str]:
        voices: dict[str, str] = {}
        if self._check_edge_tts_available():
            voices.update(EDGE_CN_VOICES)
        if self.rvc_enabled:
            for pth in _scan_rvc_models(self.rvc_model_dir):
                voices[pth.stem] = f"🎤 RVC: {pth.stem}"
        if not voices:
            voices["default"] = "默认音色"
        return voices

    @property
    def default_voice(self) -> str:
        return next(iter(self.available_voices))

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        if not self.configured or self._warmed_up or self._warming_up:
            return
        self._warming_up = True
        try:
            async with self._synthesize_lock:
                if self._check_cosyvoice_importable() and self._check_cosyvoice_downloaded():
                    await asyncio.to_thread(self._init_cosyvoice)
                    for prompt in self.prompts.list_prompts():
                        prompt_id = str(prompt.get("id", ""))
                        try:
                            prompt_wav, prompt_text = self.prompts.resolve(prompt_id)
                            await asyncio.to_thread(
                                self._prepare_cosyvoice_prompt,
                                prompt_id,
                                str(prompt_wav),
                                prompt_text,
                            )
                            logger.info(
                                "CosyVoice prompt preloaded: id=%s",
                                self._prompt_cache_id(prompt_id),
                            )
                        except ValueError:
                            logger.info(
                                "Skipping unavailable TTS prompt preload: %s",
                                prompt_id,
                            )
                        except Exception:
                            logger.warning(
                                "TTS prompt preload failed: %s",
                                prompt_id,
                                exc_info=True,
                            )
                if self.rvc_enabled:
                    await asyncio.to_thread(self._init_rvc)
                    await asyncio.to_thread(self._warm_rvc)
            self._warmed_up = True
            logger.info("TTS models preloaded")
        except Exception:
            logger.exception("TTS model preload failed; will retry on demand")
        finally:
            self._warming_up = False

    async def download_cosyvoice(self) -> None:
        from modelscope import snapshot_download

        logger.info("Downloading CosyVoice2-0.5B to %s ...", self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            snapshot_download,
            "iic/CosyVoice2-0.5B",
            local_dir=str(self.model_dir),
        )
        logger.info("CosyVoice2 model downloaded")

    def _init_cosyvoice(self) -> None:
        if self._cosyvoice_model is not None:
            return
        with self._cosyvoice_loading:
            if self._cosyvoice_model is not None:
                return
            device = _get_device(self.settings.local_tts_device)
            logger.info("Loading CosyVoice2 from %s on %s ...", self.model_dir, device)
            _ensure_vendor_on_path()
            from cosyvoice.cli.cosyvoice import CosyVoice2

            model = CosyVoice2(
                str(self.model_dir),
                load_jit=False,
                load_trt=False,
                fp16=(device != "cpu"),
            )
            self._cosyvoice_model = model
            logger.info("CosyVoice2 loaded on %s", device)

    def _init_rvc(self) -> None:
        if self._rvc_engine is not None or not self.rvc_enabled:
            return
        rvc_device = _get_device(self.settings.local_rvc_device)
        logger.info("Loading RVC on %s ...", rvc_device)
        try:
            self._rvc_engine = _RVCEngine(
                model_dir=self.rvc_model_dir,
                index_dir=self.rvc_index_dir,
                device=rvc_device,
            )
            logger.info("RVC ready")
        except ImportError:
            logger.warning("RVC 未安装，音色转换功能不可用")
            self._rvc_engine = None

    def _warm_rvc(self) -> None:
        import numpy as np

        models = _scan_rvc_models(self.rvc_model_dir)
        if self._rvc_engine is None or not models:
            return
        sample_rate = 24000
        points = np.arange(sample_rate, dtype=np.float32)
        sample = (0.1 * np.sin(2 * np.pi * 220 * points / sample_rate)).astype(
            np.float32
        )
        self._rvc_engine.convert(sample, target_model=models[0].stem)
        logger.info("RVC runtime preloaded with model %s", models[0].stem)

    # ------------------------------------------------------------------
    # 合成
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_cache_id(prompt_id: str | None) -> str:
        return f"web_agent_prompt_{prompt_id or 'default'}"

    def _prepare_cosyvoice_prompt(
        self,
        prompt_id: str | None,
        prompt_wav: str,
        prompt_text: str,
    ) -> str:
        model = self._cosyvoice_model
        if model is None:
            raise RuntimeError("CosyVoice2 尚未加载")
        path = Path(prompt_wav)
        stat = path.stat()
        cache_id = self._prompt_cache_id(prompt_id)
        cache_key = (
            str(path),
            stat.st_size,
            stat.st_mtime_ns,
            prompt_text,
        )
        metadata_id = f"__web_agent_meta__{cache_id}"
        metadata = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "text_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest(),
        }
        if self._prompt_cache_keys.get(cache_id) == cache_key:
            return cache_id
        try:
            if (
                cache_id in model.frontend.spk2info
                and model.frontend.spk2info.get(metadata_id) == metadata
            ):
                self._prompt_cache_keys[cache_id] = cache_key
                logger.info("CosyVoice prompt cache restored: id=%s", cache_id)
                return cache_id
        except (AttributeError, KeyError):
            pass
        model.add_zero_shot_spk(prompt_text, prompt_wav, cache_id)
        try:
            model.frontend.spk2info[metadata_id] = metadata
            model.save_spkinfo()
        except Exception:
            logger.warning("CosyVoice prompt cache persistence failed", exc_info=True)
        self._prompt_cache_keys[cache_id] = cache_key
        logger.info(
            "CosyVoice prompt cached: id=%s audio=%s chars=%s",
            cache_id,
            path.name,
            len(prompt_text),
        )
        return cache_id

    async def prepare_voice_prompt(self, prompt_id: str) -> bool:
        """Pre-encode a reference prompt so the first playback is fast."""
        if (
            not self._check_cosyvoice_importable()
            or not self._check_cosyvoice_downloaded()
        ):
            return False
        prompt_wav, prompt_text = self.prompts.resolve(prompt_id)
        async with self._synthesize_lock:
            if self._cosyvoice_model is None:
                await asyncio.to_thread(self._init_cosyvoice)
            await asyncio.to_thread(
                self._prepare_cosyvoice_prompt,
                prompt_id,
                str(prompt_wav),
                prompt_text,
            )
        return True

    def invalidate_voice_prompt(self, prompt_id: str) -> None:
        cache_id = self._prompt_cache_id(prompt_id)
        metadata_id = f"__web_agent_meta__{cache_id}"
        self._prompt_cache_keys.pop(cache_id, None)
        model = self._cosyvoice_model
        if model is None:
            return
        try:
            model.frontend.spk2info.pop(cache_id, None)
            model.frontend.spk2info.pop(metadata_id, None)
            model.save_spkinfo()
        except Exception:
            logger.debug(
                "Prompt cache invalidation skipped for %s",
                cache_id,
            )

    def _synth_cosyvoice(
        self,
        text: str,
        *,
        prompt_wav: str = DEFAULT_PROMPT_WAV,
        prompt_text: str = DEFAULT_PROMPT_TEXT,
        prompt_id: str | None = None,
        tts_speed: float = 1.0,
    ) -> "tuple":
        import numpy as np
        import torch

        model = self._cosyvoice_model
        assert model is not None

        if not Path(prompt_wav).is_file():
            raise ModelUpstreamError(
                f"CosyVoice2 缺少提示音频文件: {prompt_wav}"
            )

        chunks: list["numpy.ndarray"] = []
        sample_rate = int(getattr(model, "sample_rate", 24000))
        zero_shot_spk_id = self._prepare_cosyvoice_prompt(
            prompt_id,
            prompt_wav,
            prompt_text,
        )
        def generate() -> "numpy.ndarray":
            generated: list["numpy.ndarray"] = []
            with torch.inference_mode():
                results = model.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_wav,
                    zero_shot_spk_id=zero_shot_spk_id,
                    stream=False,
                    speed=tts_speed,
                )
                for result in results:
                    speech = result.get("tts_speech")
                    if speech is not None:
                        generated.append(
                            speech.detach().float().cpu().numpy().reshape(-1)
                        )
            if not generated:
                raise RuntimeError("CosyVoice2 未生成音频")
            return np.concatenate(generated)

        audio = generate()
        spoken_chars = len(
            re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text)
        )
        minimum_seconds = max(0.8, spoken_chars / 8.0)
        actual_seconds = len(audio) / max(1, sample_rate)
        if actual_seconds < minimum_seconds:
            logger.warning(
                "CosyVoice output suspiciously short: chars=%s duration=%.2fs "
                "minimum=%.2fs (not retrying)",
                spoken_chars,
                actual_seconds,
                minimum_seconds,
            )
        return audio, sample_rate

    async def _iter_cosyvoice_stream(
        self,
        text: str,
        *,
        prompt_wav: str = DEFAULT_PROMPT_WAV,
        prompt_text: str = DEFAULT_PROMPT_TEXT,
        prompt_id: str | None = None,
    ) -> AsyncIterator["tuple"]:
        """Bridge CosyVoice's synchronous native stream into asyncio."""
        import numpy as np

        model = self._cosyvoice_model
        if model is None:
            raise RuntimeError("CosyVoice2 尚未加载")
        zero_shot_spk_id = self._prepare_cosyvoice_prompt(
            prompt_id,
            prompt_wav,
            prompt_text,
        )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        sentinel = object()

        def produce() -> None:
            try:
                for output in model.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_wav,
                    zero_shot_spk_id=zero_shot_spk_id,
                    stream=True,
                ):
                    speech = output.get("tts_speech")
                    if speech is not None:
                        audio = speech.detach().float().cpu().numpy().reshape(-1)
                        loop.call_soon_threadsafe(queue.put_nowait, audio)
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=produce, name="cosyvoice-stream", daemon=True).start()
        sample_rate = int(getattr(model, "sample_rate", 24000))
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            audio = np.asarray(item, dtype=np.float32)
            if audio.size:
                yield audio, sample_rate

    @staticmethod
    def _audio_to_pcm16(audio: object) -> bytes:
        import numpy as np

        data = np.asarray(audio)
        if data.dtype == np.int16:
            return data.astype("<i2", copy=False).tobytes()
        normalized = np.clip(data.astype(np.float32), -1.0, 1.0)
        return (normalized * 32767.0).astype("<i2").tobytes()

    async def _synth_edge_tts(self, text: str, voice: str | None = None) -> "tuple":
        import edge_tts
        import soundfile as sf

        target_voice = voice if voice and voice in EDGE_CN_VOICES else _EDGE_VOICE
        communicate = edge_tts.Communicate(text, target_voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        return data, int(sr)

    @staticmethod
    def _array_to_wav(audio_data: "numpy.ndarray", sample_rate: int) -> bytes:
        import soundfile as sf

        with io.BytesIO() as buf:
            sf.write(buf, audio_data, sample_rate, format="WAV")
            return buf.getvalue()

    @staticmethod
    def _wav_to_mp3(wav_bytes: bytes) -> bytes:
        from pydub import AudioSegment

        try:
            import imageio_ffmpeg
            AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        out = io.BytesIO()
        seg.export(out, format="mp3", bitrate="128k")
        return out.getvalue()

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.readframes(w.getnframes())

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "mp3",
        engine: str | None = None,
        rvc_enabled: bool | None = None,
        rvc_model: str | None = None,
        rvc_index: str | None = None,
    ) -> AsyncIterator[bytes]:
        if not self.configured:
            raise ModelUpstreamError("TTS 服务未配置")

        text = text.strip()
        if not text:
            raise ValueError("文本不能为空")
        if len(text) > MAX_TTS_CHARS:
            text = text[:MAX_TTS_CHARS]

        target_engine = (engine or self._engine_type).strip().lower()

        is_edge_voice = voice is not None and voice in EDGE_CN_VOICES
        is_rvc_voice = (
            voice is not None
            and voice != "default"
            and voice not in EDGE_CN_VOICES
            and (self.rvc_model_dir / f"{voice}.pth").is_file()
        )
        selected_rvc_model = rvc_model or (voice if is_rvc_voice else None)
        use_rvc = (
            self.rvc_enabled
            and bool(selected_rvc_model)
            and (rvc_enabled if rvc_enabled is not None else is_rvc_voice)
        )

        if is_edge_voice:
            target_engine = "edge_tts"
        elif is_rvc_voice and not engine:
            # Older frontend builds do not send the base-engine field. Prefer
            # the fast Edge base for RVC instead of blocking on autogressive TTS.
            target_engine = "edge_tts"

        async with self._synthesize_lock:
            cosy_available = (
                self._check_cosyvoice_importable()
                and self._check_cosyvoice_downloaded()
            )

            use_cosy = target_engine == "cosyvoice" and cosy_available

            if target_engine == "cosyvoice" and not cosy_available:
                logger.warning(
                    "CosyVoice2 不可用，fallback 到 edge_tts "
                    "(import=%s, downloaded=%s)",
                    self._check_cosyvoice_importable(),
                    self._check_cosyvoice_downloaded(),
                )

            if use_cosy:
                if self._cosyvoice_model is None:
                    await asyncio.to_thread(self._init_cosyvoice)
                if use_rvc and self._rvc_engine is None:
                    await asyncio.to_thread(self._init_rvc)
                self._engine_type = "cosyvoice"
                audio, sample_rate = await asyncio.to_thread(
                    self._synth_cosyvoice, text
                )
                if self._rvc_engine is not None and use_rvc:
                    audio, sample_rate = await asyncio.to_thread(
                        self._rvc_engine.convert,
                        audio,
                        target_model=selected_rvc_model,
                        index_name=rvc_index,
                    )
                wav_bytes = await asyncio.to_thread(
                    self._array_to_wav, audio, sample_rate
                )
            else:
                if self._check_edge_tts_available():
                    synth_voice = (
                        voice if voice and voice in EDGE_CN_VOICES else None
                    )
                    try:
                        audio, sample_rate = await self._synth_edge_tts(
                            text, voice=synth_voice
                        )
                        self._engine_type = "edge_tts"
                    except Exception as exc:
                        if not cosy_available:
                            raise ModelUpstreamError(
                                "Edge TTS 调用失败，且本地 CosyVoice2 不可用"
                            ) from exc
                        logger.warning(
                            "Edge TTS 不可用，fallback 到本地 CosyVoice2: %s",
                            exc,
                        )
                        if self._cosyvoice_model is None:
                            await asyncio.to_thread(self._init_cosyvoice)
                        audio, sample_rate = await asyncio.to_thread(
                            self._synth_cosyvoice, text
                        )
                        self._engine_type = "cosyvoice"
                elif cosy_available:
                    if self._cosyvoice_model is None:
                        await asyncio.to_thread(self._init_cosyvoice)
                    audio, sample_rate = await asyncio.to_thread(
                        self._synth_cosyvoice, text
                    )
                    self._engine_type = "cosyvoice"
                else:
                    raise ModelUpstreamError(
                        "TTS 引擎均不可用 (CosyVoice2 + edge-tts)"
                    )
                if use_rvc and self._rvc_engine is None:
                    await asyncio.to_thread(self._init_rvc)
                if self._rvc_engine is not None and use_rvc:
                    audio, sample_rate = await asyncio.to_thread(
                        self._rvc_engine.convert,
                        audio,
                        target_model=selected_rvc_model,
                        index_name=rvc_index,
                    )
                wav_bytes = await asyncio.to_thread(
                    self._array_to_wav, audio, sample_rate
                )

        if response_format == "wav":
            output = wav_bytes
        elif response_format == "mp3":
            output = await asyncio.to_thread(self._wav_to_mp3, wav_bytes)
        elif response_format == "pcm":
            output = await asyncio.to_thread(self._wav_to_pcm, wav_bytes)
        else:
            output = wav_bytes

        chunk_size = 4096
        for i in range(0, len(output), chunk_size):
            yield output[i : i + chunk_size]

    async def synthesize_segments(
        self,
        text: str,
        *,
        voice: str | None = None,
        engine: str | None = None,
        rvc_enabled: bool | None = None,
        rvc_model: str | None = None,
        rvc_index: str | None = None,
    ) -> AsyncIterator[tuple[str, bytes, str]]:
        if len(text) > MAX_TTS_CHARS:
            text = text[:MAX_TTS_CHARS]
        for segment in _split_tts_text(text):
            chunks: list[bytes] = []
            async for chunk in self.synthesize(
                segment,
                voice=voice,
                response_format="wav",
                engine=engine,
                rvc_enabled=rvc_enabled,
                rvc_model=rvc_model,
                rvc_index=rvc_index,
            ):
                chunks.append(chunk)
            yield segment, b"".join(chunks), self._engine_type

    async def synthesize_pcm_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        engine: str | None = None,
        rvc_enabled: bool | None = None,
        rvc_model: str | None = None,
        rvc_index: str | None = None,
        prompt_id: str | None = None,
        rvc_index_rate: float = 0.6,
        rvc_protect: float = 0.4,
        rvc_rms_mix_rate: float = 0.25,
        rvc_f0_up_key: int = 0,
        tts_speed: float = 1.0,
    ) -> AsyncIterator[tuple[str, int, bytes, str]]:
        """Yield adaptively chunked audio as mono signed 16-bit PCM."""
        if not self.configured:
            raise ModelUpstreamError("TTS 服务未配置")

        text = _normalize_spoken_text(text.strip())[:MAX_TTS_CHARS]
        if not text:
            raise ValueError("文本不能为空")

        target_engine = (engine or self._engine_type).strip().lower()
        is_edge_voice = voice is not None and voice in EDGE_CN_VOICES
        is_rvc_voice = (
            voice is not None
            and voice != "default"
            and voice not in EDGE_CN_VOICES
            and (self.rvc_model_dir / f"{voice}.pth").is_file()
        )
        selected_rvc_model = rvc_model or (voice if is_rvc_voice else None)
        use_rvc = (
            self.rvc_enabled
            and bool(selected_rvc_model)
            and (rvc_enabled if rvc_enabled is not None else is_rvc_voice)
        )
        if is_edge_voice:
            target_engine = "edge_tts"
        elif is_rvc_voice and not engine:
            target_engine = "edge_tts"

        prompt_wav = Path(DEFAULT_PROMPT_WAV)
        prompt_text = DEFAULT_PROMPT_TEXT
        prompt_resolved = False

        def resolve_prompt() -> tuple[Path, str]:
            nonlocal prompt_resolved, prompt_wav, prompt_text
            if not prompt_resolved:
                prompt_wav, prompt_text = self.prompts.resolve(prompt_id)
                prompt_resolved = True
            return prompt_wav, prompt_text

        index_rate = max(0.0, min(1.0, float(rvc_index_rate)))
        protect = max(0.0, min(0.5, float(rvc_protect)))
        rms_mix_rate = max(0.0, min(1.0, float(rvc_rms_mix_rate)))
        f0_up_key = max(-12, min(12, int(rvc_f0_up_key)))
        cosy_speed = max(0.7, min(1.3, float(tts_speed)))

        async with self._synthesize_lock:
            cosy_available = (
                self._check_cosyvoice_importable()
                and self._check_cosyvoice_downloaded()
            )
            edge_available = self._check_edge_tts_available()

            if use_rvc and self._rvc_engine is None:
                await asyncio.to_thread(self._init_rvc)

            if not edge_available and not cosy_available:
                raise ModelUpstreamError(
                    "TTS 引擎均不可用 (CosyVoice2 + edge-tts)"
                )
            if target_engine == "cosyvoice" and not cosy_available:
                detail = self._cosyvoice_import_error or "模型文件不完整"
                raise ModelUpstreamError(
                    f"CosyVoice2 不可用，已停止 Edge TTS 静默降级：{detail}"
                )

            async def source_segments() -> AsyncIterator[tuple[str, object, int, str]]:
                use_cosy = target_engine == "cosyvoice" and cosy_available
                if use_cosy and self._cosyvoice_model is None:
                    await asyncio.to_thread(self._init_cosyvoice)
                if use_cosy:
                    active_prompt_wav, active_prompt_text = resolve_prompt()
                    await asyncio.to_thread(
                        self._prepare_cosyvoice_prompt,
                        prompt_id,
                        str(active_prompt_wav),
                        active_prompt_text,
                    )
                if not use_cosy and edge_available:
                    synth_voice = (
                        voice if voice and voice in EDGE_CN_VOICES else None
                    )

                sentence_segments = _split_tts_sentences(text)
                if use_cosy:
                    sentence_segments = _group_sentence_segments(
                        sentence_segments,
                    )
                    logger.info(
                        "TTS sentence grouping: groups=%s",
                        [len(item) for item in sentence_segments],
                    )

                for segment in sentence_segments:
                    started = time.perf_counter()
                    if use_cosy:
                        audio, sample_rate = await asyncio.to_thread(
                            self._synth_cosyvoice,
                            segment,
                            prompt_wav=str(prompt_wav),
                            prompt_text=prompt_text,
                            prompt_id=prompt_id,
                            tts_speed=cosy_speed,
                        )
                        self._engine_type = "cosyvoice"
                    elif edge_available:
                        try:
                            audio, sample_rate = await self._synth_edge_tts(
                                segment, voice=synth_voice
                            )
                            self._engine_type = "edge_tts"
                        except Exception:
                            if not cosy_available:
                                raise
                            if self._cosyvoice_model is None:
                                await asyncio.to_thread(self._init_cosyvoice)
                            active_prompt_wav, active_prompt_text = resolve_prompt()
                            audio, sample_rate = await asyncio.to_thread(
                                self._synth_cosyvoice,
                                segment,
                                prompt_wav=str(active_prompt_wav),
                                prompt_text=active_prompt_text,
                                prompt_id=prompt_id,
                                tts_speed=cosy_speed,
                            )
                            self._engine_type = "cosyvoice"
                    else:
                        if self._cosyvoice_model is None:
                            await asyncio.to_thread(self._init_cosyvoice)
                        active_prompt_wav, active_prompt_text = resolve_prompt()
                        audio, sample_rate = await asyncio.to_thread(
                            self._synth_cosyvoice,
                            segment,
                            prompt_wav=str(active_prompt_wav),
                            prompt_text=active_prompt_text,
                            prompt_id=prompt_id,
                            tts_speed=cosy_speed,
                        )
                        self._engine_type = "cosyvoice"

                    yield segment, audio, sample_rate, self._engine_type
                    elapsed = time.perf_counter() - started
                    audio_duration = max(len(audio) / sample_rate, 0.1)
                    ratio = elapsed / audio_duration
                    logger.debug(
                        "TTS segment completed: chars=%s duration=%.2fs rtf=%.2f",
                        len(segment),
                        audio_duration,
                        ratio,
                    )

            if not (use_rvc and self._rvc_engine is not None):
                async for segment, audio, sample_rate, used_engine in source_segments():
                    yield (
                        segment,
                        int(sample_rate),
                        self._audio_to_pcm16(audio),
                        used_engine,
                    )
                return

            import numpy as np

            stream_end = object()
            # Keep the pipeline immediate: one source segment and one converted
            # segment may be in flight, without multi-segment buffering.
            source_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
            output_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)

            async def produce_source() -> None:
                try:
                    async for item in source_segments():
                        await source_queue.put(item)
                except Exception as exc:
                    await source_queue.put(exc)
                finally:
                    await source_queue.put(stream_end)

            async def convert_rvc() -> None:
                try:
                    while True:
                        item = await source_queue.get()
                        if item is stream_end:
                            await output_queue.put(stream_end)
                            break
                        if isinstance(item, Exception):
                            raise item
                        segment, audio, sample_rate, used_engine = item
                        source = np.asarray(audio, dtype=np.float32)
                        converted, rvc_sample_rate = await asyncio.to_thread(
                            self._rvc_engine.convert,
                            source,
                            target_model=selected_rvc_model,
                            index_name=rvc_index,
                            index_rate=index_rate,
                            protect=protect,
                            rms_mix_rate=rms_mix_rate,
                            f0_up_key=f0_up_key,
                        )
                        await output_queue.put(
                            (
                                segment,
                                rvc_sample_rate,
                                self._audio_to_pcm16(converted),
                                used_engine,
                            )
                        )
                except Exception as exc:
                    await output_queue.put(exc)

            producer_task = asyncio.create_task(produce_source())
            converter_task = asyncio.create_task(convert_rvc())
            try:
                while True:
                    item = await output_queue.get()
                    if item is stream_end:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
                await asyncio.gather(producer_task, converter_task)
            finally:
                for task in (producer_task, converter_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    producer_task,
                    converter_task,
                    return_exceptions=True,
                )


# ------------------------------------------------------------------
# RVC 引擎
# ------------------------------------------------------------------

class _RVCEngine:
    def __init__(
        self,
        *,
        model_dir: Path,
        index_dir: Path,
        device: str,
    ) -> None:
        self.model_dir = model_dir
        self.index_dir = index_dir
        self.device = device
        self._vc_class = None
        self._vc: object | None = None
        self._settings: object | None = None
        self._loaded_model_key: tuple[str, int, int] | None = None
        self._init_backend()

    def _init_backend(self) -> None:
        _ensure_vendor_on_path()
        # CosyVoice vendors Matcha-TTS, which also exposes a top-level
        # ``configs`` package. RVC imports its own ``configs.config`` lazily,
        # so discard the cached Matcha namespace before loading RVC.
        for module_name in tuple(sys.modules):
            if module_name == "configs" or module_name.startswith("configs."):
                sys.modules.pop(module_name, None)
        previous_cwd = Path.cwd()
        try:
            # The vendored RVC runtime loads locale and CUDA assets relative to
            # its own root. Keep the process cwd stable outside initialization.
            os.chdir(RVC_VENDOR_DIR)
            try:
                from infer.vc.modules import VC
                self._vc_class = VC
            except ImportError as exc:
                raise ImportError(
                    "RVC backend not found. Ensure vendor/RVC is in the project root."
                ) from exc
        finally:
            os.chdir(previous_cwd)

    def _get_vc(self, pth_path: Path) -> object:
        from types import SimpleNamespace

        if self._vc is None:
            self._settings = SimpleNamespace(
                device=self.device,
                is_half=(self.device != "cpu"),
                x_pad=3,
                x_query=10,
                x_center=60,
                x_max=65,
            )
            self._vc = self._vc_class(self._settings)

        stat = pth_path.stat()
        model_key = (str(pth_path), stat.st_size, stat.st_mtime_ns)
        if model_key != self._loaded_model_key:
            self._vc.get_vc(pth_path.name)
            self._loaded_model_key = model_key
        return self._vc

    def _resolve_model(self, target_model: str) -> tuple[Path, Path | None]:
        pth_path = self.model_dir / f"{target_model}.pth"
        if not pth_path.exists():
            all_pths = list(self.model_dir.glob("*.pth"))
            if not all_pths:
                raise FileNotFoundError(
                    f"No .pth model in {self.model_dir}"
                )
            pth_path = all_pths[0]

        stem = pth_path.stem
        index_path = None
        if self.index_dir.exists():
            candidates = sorted(
                self.index_dir.glob(f"{stem}*.index"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                index_path = candidates[0]
        return pth_path, index_path

    def convert(
        self,
        source_wav,
        *,
        target_model: str = "",
        index_name: str | None = None,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.6,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.15,
        protect: float = 0.4,
    ):
        import tempfile

        import numpy as np
        import soundfile as sf

        target = target_model or "model"
        pth_path, index_path = self._resolve_model(target)
        if index_name:
            requested_index = self.index_dir / Path(index_name).name
            if not requested_index.is_file():
                raise FileNotFoundError(f"RVC index not found: {requested_index}")
            index_path = requested_index
        logger.info(
            "RVC converting with model=%s index=%s",
            pth_path.name,
            index_path.name if index_path else "(none)",
        )

        os.environ["weight_root"] = str(self.model_dir)
        if self.index_dir.exists():
            os.environ["index_root"] = str(self.index_dir)
            os.environ["outside_index_root"] = str(self.index_dir)
        os.environ["rmvpe_root"] = str(RVC_VENDOR_DIR / "assets" / "rmvpe")

        vc = self._get_vc(pth_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, source_wav.astype(np.float32), 24000)

        try:
            status, result = vc.vc_single(
                sid=0,
                input_audio_path=tmp_path,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                file_index=str(index_path) if index_path else "",
                index_rate=index_rate,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )
            if result is None or result[1] is None:
                raise RuntimeError(f"RVC 转换失败: {status}")
            return result[1], result[0]
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
