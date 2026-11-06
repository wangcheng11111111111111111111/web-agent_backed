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


class ConfigError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ConfigError(f"缺少必需的环境变量：{name}")
    return value.strip()


def _allow_empty(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ConfigError(f"缺少环境变量：{name}")
    return value.strip()


def _int(name: str) -> int:
    try:
        return int(_required(name))
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是整数") from exc


def _float(name: str) -> float:
    try:
        return float(_required(name))
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是数字") from exc


def _bool(name: str) -> bool:
    value = _required(name).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"环境变量 {name} 必须是布尔值")


def _str(name: str) -> str:
    return _required(name)


def _url(name: str) -> str:
    return _str(name).rstrip("/")


def _list(name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in _str(name).split(",") if item.strip())
    if not values:
        raise ConfigError(f"环境变量 {name} 不能为空")
    return values


@dataclass(frozen=True)
class Settings:
    app_env: str = field(default_factory=lambda: _str("APP_ENV"))
    api_host: str = field(default_factory=lambda: _str("API_HOST"))
    api_port: int = field(default_factory=lambda: _int("API_PORT"))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _list("CORS_ORIGINS")
    )
    global_client_id: str = field(
        default_factory=lambda: _str("GLOBAL_CLIENT_ID")
    )
    global_tenant_id: str = field(
        default_factory=lambda: _str("GLOBAL_TENANT_ID")
    )
    global_user_id: str = field(
        default_factory=lambda: _str("GLOBAL_USER_ID")
    )

    llm_api_key: str = field(
        default_factory=lambda: _allow_empty("LLM_API_KEY"), repr=False
    )
    llm_base_url: str = field(default_factory=lambda: _url("LLM_BASE_URL"))
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL"))
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT"))

    pg_host: str = field(default_factory=lambda: _str("PG_HOST"))
    pg_port: int = field(default_factory=lambda: _int("PG_PORT"))
    pg_database: str = field(default_factory=lambda: _str("PG_DATABASE"))
    pg_username: str = field(default_factory=lambda: _str("PG_USERNAME"))
    pg_password: str = field(
        default_factory=lambda: _allow_empty("PG_PASSWORD"),
        repr=False,
    )
    pg_sslmode: str = field(default_factory=lambda: _str("PG_SSLMODE"))

    redis_host: str = field(default_factory=lambda: _str("REDIS_HOST"))
    redis_port: int = field(default_factory=lambda: _int("REDIS_PORT"))
    redis_db: int = field(default_factory=lambda: _int("REDIS_DB"))
    redis_password: str = field(
        default_factory=lambda: _allow_empty("REDIS_PASSWORD"),
        repr=False,
    )
    redis_enabled: bool = field(default_factory=lambda: _bool("REDIS_ENABLED"))
    session_ttl_seconds: int = field(
        default_factory=lambda: _int("SESSION_TTL_SECONDS")
    )
    short_term_message_limit: int = field(
        default_factory=lambda: _int("SHORT_TERM_MESSAGE_LIMIT")
    )
    conversation_recall_enabled: bool = field(
        default_factory=lambda: _bool("CONVERSATION_RECALL_ENABLED")
    )
    conversation_recall_scan: int = field(
        default_factory=lambda: _int("CONVERSATION_RECALL_SCAN")
    )
    conversation_recall_limit: int = field(
        default_factory=lambda: _int("CONVERSATION_RECALL_LIMIT")
    )
    conversation_recall_max_chars: int = field(
        default_factory=lambda: _int("CONVERSATION_RECALL_MAX_CHARS")
    )
    background_media_root: str = field(
        default_factory=lambda: _str("BACKGROUND_MEDIA_ROOT")
    )
    max_background_video_mb: int = field(
        default_factory=lambda: _int("MAX_BACKGROUND_VIDEO_MB")
    )
    brand_media_root: str = field(
        default_factory=lambda: _str("BRAND_MEDIA_ROOT")
    )

    qdrant_host: str = field(default_factory=lambda: _str("QDRANT_HOST"))
    qdrant_port: int = field(default_factory=lambda: _int("QDRANT_PORT"))
    qdrant_collection: str = field(
        default_factory=lambda: _str("QDRANT_COLLECTION")
    )
    qdrant_api_key: str = field(
        default_factory=lambda: _allow_empty("QDRANT_API_KEY"), repr=False
    )
    qdrant_timeout: float = field(default_factory=lambda: _float("QDRANT_TIMEOUT"))

    embedding_base_url: str = field(
        default_factory=lambda: _url("EMBEDDING_BASE_URL")
    )
    embedding_api_key: str = field(
        default_factory=lambda: _allow_empty("EMBEDDING_API_KEY"),
        repr=False,
    )
    embedding_model: str = field(default_factory=lambda: _str("EMBEDDING_MODEL"))
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM"))
    embedding_timeout: float = field(
        default_factory=lambda: _float("EMBEDDING_TIMEOUT")
    )

    rag_top_k: int = field(default_factory=lambda: _int("RAG_TOP_K"))
    rag_score_threshold: float = field(
        default_factory=lambda: _float("RAG_SCORE_THRESHOLD")
    )
    rag_chunk_size: int = field(
        default_factory=lambda: _int("RAG_CHUNK_SIZE")
    )
    rag_chunk_overlap: int = field(
        default_factory=lambda: _int("RAG_CHUNK_OVERLAP")
    )
    max_upload_mb: int = field(
        default_factory=lambda: _int("MAX_UPLOAD_MB")
    )

    memory_enabled: bool = field(default_factory=lambda: _bool("MEMORY_ENABLED"))
    memory_collection: str = field(
        default_factory=lambda: _str("MEMORY_COLLECTION")
    )
    memory_top_k: int = field(default_factory=lambda: _int("MEMORY_TOP_K"))
    memory_retrieval_threshold: float = field(
        default_factory=lambda: _float("MEMORY_RETRIEVAL_THRESHOLD")
    )
    memory_dedup_threshold: float = field(
        default_factory=lambda: _float("MEMORY_DEDUP_THRESHOLD")
    )
    memory_correction_threshold: float = field(
        default_factory=lambda: _float("MEMORY_CORRECTION_THRESHOLD")
    )
    memory_recency_days: int = field(
        default_factory=lambda: _int("MEMORY_RECENCY_DAYS")
    )
    memory_min_confidence: float = field(
        default_factory=lambda: _float("MEMORY_MIN_CONFIDENCE")
    )
    memory_max_facts: int = field(default_factory=lambda: _int("MEMORY_MAX_FACTS"))
    memory_min_content_chars: int = field(
        default_factory=lambda: _int("MEMORY_MIN_CONTENT_CHARS")
    )
    memory_max_content_chars: int = field(
        default_factory=lambda: _int("MEMORY_MAX_CONTENT_CHARS")
    )

    web_search_enabled: bool = field(
        default_factory=lambda: _bool("WEB_SEARCH_ENABLED")
    )
    web_search_top_k: int = field(
        default_factory=lambda: _int("WEB_SEARCH_TOP_K")
    )
    web_search_timeout: float = field(
        default_factory=lambda: _float("WEB_SEARCH_TIMEOUT")
    )

    web_crawl_enabled: bool = field(
        default_factory=lambda: _bool("WEB_CRAWL_ENABLED")
    )
    web_crawl_timeout: float = field(
        default_factory=lambda: _float("WEB_CRAWL_TIMEOUT")
    )
    web_crawl_max_chars: int = field(
        default_factory=lambda: _int("WEB_CRAWL_MAX_CHARS")
    )

    tts_enabled: bool = field(default_factory=lambda: _bool("TTS_ENABLED"))
    local_tts_engine: str = field(
        default_factory=lambda: _str("LOCAL_TTS_ENGINE").lower()
    )
    local_tts_model_dir: str = field(
        default_factory=lambda: _str("LOCAL_TTS_MODEL_DIR")
    )
    local_tts_device: str = field(
        default_factory=lambda: _str("LOCAL_TTS_DEVICE").lower()
    )
    local_rvc_model_dir: str = field(
        default_factory=lambda: _str("LOCAL_RVC_MODEL_DIR")
    )
    local_rvc_index_dir: str = field(
        default_factory=lambda: _str("LOCAL_RVC_INDEX_DIR")
    )
    local_rvc_device: str = field(
        default_factory=lambda: _str("LOCAL_RVC_DEVICE").lower()
    )

    stt_enabled: bool = field(default_factory=lambda: _bool("STT_ENABLED"))
    stt_device: str = field(default_factory=lambda: _str("STT_DEVICE").lower())
    stt_whisper_model: str = field(
        default_factory=lambda: _str("STT_WHISPER_MODEL").lower()
    )
    stt_base_url: str = field(default_factory=lambda: _url("STT_BASE_URL"))
    stt_api_key: str = field(
        default_factory=lambda: _allow_empty("STT_API_KEY"),
        repr=False,
    )
    stt_model: str = field(default_factory=lambda: _str("STT_MODEL"))
    stt_timeout: float = field(default_factory=lambda: _float("STT_TIMEOUT"))

    @property
    def database_dsn(self) -> str:
        return (
            f"postgresql://{quote_plus(self.pg_username)}:"
            f"{quote_plus(self.pg_password)}@{self.pg_host}:{self.pg_port}/"
            f"{quote_plus(self.pg_database)}"
            f"?sslmode={self.pg_sslmode}"
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
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def embeddings_configured(self) -> bool:
        return bool(
            self.embedding_api_key
            and self.embedding_base_url
            and self.embedding_model
        )

    @property
    def remote_stt_configured(self) -> bool:
        return bool(
            self.stt_enabled
            and self.stt_api_key
            and self.stt_base_url
            and self.stt_model
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
