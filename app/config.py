from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_PERSONALITY_PROMPT: str = "你是一个严谨、友好、简洁的中文 AI 助手。"

SYSTEM_RULES_PROMPT: str = (
    "当提供知识库片段时，优先依据片段回答，并在相关陈述后用 "
    "[来源: 文件名] 标注来源；不要编造片段中没有的事实。"
    "当提供联网搜索结果时，结合最新信息回答，并在相关陈述后用 "
    "[来源: 标题] 标注。知识库和联网结果冲突时，优先联网结果（通常更新）。"
    "如果知识库片段不足或与联网结果均无相关内容，可以结合通用知识回答，"
    "并明确说明。"
    "工具结果由后端实际执行，必须将其视为可靠事实。"
    "用户长期记忆是与当前用户相关的跨会话事实；只在与问题相关时使用，"
    "不要向用户逐条复述记忆内容。"
)

DEFAULT_SYSTEM_PROMPT: str = DEFAULT_PERSONALITY_PROMPT + SYSTEM_RULES_PROMPT


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _int("API_PORT", 8000))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip()
            for item in os.getenv(
                "CORS_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173",
            ).split(",")
            if item.strip()
        )
    )

    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "").strip(),
        repr=False,
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://api.deepseek.com/v1"
        ).rstrip("/")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )
    llm_timeout: float = field(
        default_factory=lambda: _float("LLM_TIMEOUT", 60.0)
    )

    pg_host: str = field(default_factory=lambda: os.getenv("PG_HOST", "127.0.0.1"))
    pg_port: int = field(default_factory=lambda: _int("PG_PORT", 5432))
    pg_database: str = field(
        default_factory=lambda: os.getenv("PG_DATABASE", "ai_assistant")
    )
    pg_username: str = field(
        default_factory=lambda: os.getenv("PG_USERNAME", "ai_assistant")
    )
    pg_password: str = field(
        default_factory=lambda: os.getenv("PG_PASSWORD", ""),
        repr=False,
    )
    pg_sslmode: str = field(
        default_factory=lambda: os.getenv("PG_SSLMODE", "prefer")
    )

    redis_host: str = field(
        default_factory=lambda: os.getenv("REDIS_HOST", "127.0.0.1")
    )
    redis_port: int = field(default_factory=lambda: _int("REDIS_PORT", 6379))
    redis_db: int = field(default_factory=lambda: _int("REDIS_DB", 0))
    redis_password: str = field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD", ""),
        repr=False,
    )
    redis_enabled: bool = field(
        default_factory=lambda: _bool("REDIS_ENABLED", True)
    )
    session_ttl_seconds: int = field(
        default_factory=lambda: _int("SESSION_TTL_SECONDS", 604800)
    )
    short_term_message_limit: int = field(
        default_factory=lambda: _int("SHORT_TERM_MESSAGE_LIMIT", 40)
    )
    conversation_recall_enabled: bool = field(
        default_factory=lambda: _bool("CONVERSATION_RECALL_ENABLED", True)
    )
    conversation_recall_scan: int = field(
        default_factory=lambda: _int("CONVERSATION_RECALL_SCAN", 200)
    )
    conversation_recall_limit: int = field(
        default_factory=lambda: _int("CONVERSATION_RECALL_LIMIT", 6)
    )
    conversation_recall_max_chars: int = field(
        default_factory=lambda: _int("CONVERSATION_RECALL_MAX_CHARS", 2400)
    )
    background_media_root: str = field(
        default_factory=lambda: os.getenv(
            "BACKGROUND_MEDIA_ROOT",
            str(BACKEND_DIR / "data" / "backgrounds"),
        )
    )
    max_background_video_mb: int = field(
        default_factory=lambda: _int("MAX_BACKGROUND_VIDEO_MB", 10240)
    )
    brand_media_root: str = field(
        default_factory=lambda: os.getenv(
            "BRAND_MEDIA_ROOT",
            str(BACKEND_DIR / "data" / "branding"),
        )
    )

    qdrant_host: str = field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "127.0.0.1")
    )
    qdrant_port: int = field(default_factory=lambda: _int("QDRANT_PORT", 6333))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "knowledge_base")
    )
    qdrant_api_key: str = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", "").strip(),
        repr=False,
    )
    qdrant_timeout: float = field(
        default_factory=lambda: _float("QDRANT_TIMEOUT", 10.0)
    )

    embedding_base_url: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"
        ).rstrip("/")
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "").strip(),
        repr=False,
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    embedding_dim: int = field(
        default_factory=lambda: _int("EMBEDDING_DIM", 1024)
    )
    embedding_timeout: float = field(
        default_factory=lambda: _float("EMBEDDING_TIMEOUT", 30.0)
    )

    rag_top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 5))
    rag_score_threshold: float = field(
        default_factory=lambda: _float("RAG_SCORE_THRESHOLD", 0.45)
    )
    rag_chunk_size: int = field(
        default_factory=lambda: _int("RAG_CHUNK_SIZE", 900)
    )
    rag_chunk_overlap: int = field(
        default_factory=lambda: _int("RAG_CHUNK_OVERLAP", 120)
    )
    max_upload_mb: int = field(
        default_factory=lambda: _int("MAX_UPLOAD_MB", 10)
    )

    memory_enabled: bool = field(
        default_factory=lambda: _bool("MEMORY_ENABLED", True)
    )
    memory_collection: str = field(
        default_factory=lambda: os.getenv("MEMORY_COLLECTION", "user_memory")
    )
    memory_top_k: int = field(
        default_factory=lambda: _int("MEMORY_TOP_K", 5)
    )
    memory_retrieval_threshold: float = field(
        default_factory=lambda: _float("MEMORY_RETRIEVAL_THRESHOLD", 0.35)
    )
    memory_dedup_threshold: float = field(
        default_factory=lambda: _float("MEMORY_DEDUP_THRESHOLD", 0.92)
    )
    memory_correction_threshold: float = field(
        default_factory=lambda: _float("MEMORY_CORRECTION_THRESHOLD", 0.68)
    )
    memory_recency_days: int = field(
        default_factory=lambda: _int("MEMORY_RECENCY_DAYS", 90)
    )
    memory_min_confidence: float = field(
        default_factory=lambda: _float("MEMORY_MIN_CONFIDENCE", 0.60)
    )
    memory_max_facts: int = field(
        default_factory=lambda: _int("MEMORY_MAX_FACTS", 5)
    )
    memory_min_content_chars: int = field(
        default_factory=lambda: _int("MEMORY_MIN_CONTENT_CHARS", 4)
    )
    memory_max_content_chars: int = field(
        default_factory=lambda: _int("MEMORY_MAX_CONTENT_CHARS", 800)
    )

    web_search_enabled: bool = field(
        default_factory=lambda: _bool("WEB_SEARCH_ENABLED", True)
    )
    web_search_provider: str = field(
        default_factory=lambda: os.getenv(
            "WEB_SEARCH_PROVIDER", "bing"
        ).strip().lower()
    )
    web_search_api_key: str = field(
        default_factory=lambda: os.getenv("WEB_SEARCH_API_KEY", "").strip(),
        repr=False,
    )
    web_search_top_k: int = field(
        default_factory=lambda: _int("WEB_SEARCH_TOP_K", 5)
    )
    web_search_timeout: float = field(
        default_factory=lambda: _float("WEB_SEARCH_TIMEOUT", 15.0)
    )

    web_crawl_enabled: bool = field(
        default_factory=lambda: _bool("WEB_CRAWL_ENABLED", True)
    )
    web_crawl_timeout: float = field(
        default_factory=lambda: _float("WEB_CRAWL_TIMEOUT", 20.0)
    )
    web_crawl_max_chars: int = field(
        default_factory=lambda: _int("WEB_CRAWL_MAX_CHARS", 4000)
    )

    tts_enabled: bool = field(
        default_factory=lambda: _bool("TTS_ENABLED", False)
    )
    local_tts_engine: str = field(
        default_factory=lambda: os.getenv("LOCAL_TTS_ENGINE", "auto").strip().lower()
    )
    local_tts_model_dir: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_TTS_MODEL_DIR",
            str(BACKEND_DIR / "models" / "cosyvoice2"),
        )
    )
    local_tts_device: str = field(
        default_factory=lambda: os.getenv("LOCAL_TTS_DEVICE", "cuda").strip().lower()
    )
    local_rvc_model_dir: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_RVC_MODEL_DIR",
            str(BACKEND_DIR / "models" / "rvc" / "weights"),
        )
    )
    local_rvc_index_dir: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_RVC_INDEX_DIR",
            str(BACKEND_DIR / "models" / "rvc" / "indexes"),
        )
    )
    local_rvc_device: str = field(
        default_factory=lambda: os.getenv("LOCAL_RVC_DEVICE", "cuda").strip().lower()
    )

    stt_enabled: bool = field(
        default_factory=lambda: _bool("STT_ENABLED", False)
    )
    stt_device: str = field(
        default_factory=lambda: os.getenv(
            "STT_DEVICE", "auto"
        ).strip().lower()
    )
    stt_whisper_model: str = field(
        default_factory=lambda: os.getenv(
            "STT_WHISPER_MODEL", "small"
        ).strip().lower()
    )
    stt_base_url: str = field(
        default_factory=lambda: os.getenv(
            "STT_BASE_URL", "https://api.siliconflow.cn/v1"
        ).rstrip("/")
    )
    stt_api_key: str = field(
        default_factory=lambda: os.getenv(
            "STT_API_KEY", os.getenv("EMBEDDING_API_KEY", "")
        ).strip(),
        repr=False,
    )
    stt_model: str = field(
        default_factory=lambda: os.getenv(
            "STT_MODEL", "FunAudioLLM/SenseVoiceSmall"
        )
    )
    stt_timeout: float = field(
        default_factory=lambda: _float("STT_TIMEOUT", 30.0)
    )

    @property
    def database_dsn(self) -> str:
        sslmode = self.pg_sslmode or "prefer"
        return (
            f"postgresql://{quote_plus(self.pg_username)}:"
            f"{quote_plus(self.pg_password)}@{self.pg_host}:{self.pg_port}/"
            f"{quote_plus(self.pg_database)}"
            f"?sslmode={sslmode}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def background_media_dir(self) -> Path:
        path = Path(self.background_media_root)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return path.resolve()

    @property
    def brand_media_dir(self) -> Path:
        path = Path(self.brand_media_root)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return path.resolve()

    @property
    def model_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def embeddings_configured(self) -> bool:
        return bool(self.embedding_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
