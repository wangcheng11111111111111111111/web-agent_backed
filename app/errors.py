from __future__ import annotations


class AppError(Exception):
    code = "application_error"
    status_code = 500
    public_message = "服务暂时不可用"

    def __init__(self, message: str | None = None) -> None:
        self.public_message = message or self.public_message
        super().__init__(self.public_message)


class DatabaseUnavailableError(AppError):
    code = "database_unavailable"
    status_code = 503
    public_message = "后端暂时无法连接会话数据库，请检查 PostgreSQL 服务和配置"


class SessionNotFoundError(AppError):
    code = "session_not_found"
    status_code = 404
    public_message = "会话不存在或已被删除"


class ModelConfigError(AppError):
    code = "model_key_missing"
    status_code = 503
    public_message = "后端未配置模型密钥，请在 backend/.env 中设置 LLM_API_KEY"


class ModelTimeoutError(AppError):
    code = "model_timeout"
    status_code = 504
    public_message = "模型响应超时，请稍后重试"


class ModelUpstreamError(AppError):
    code = "model_upstream_error"
    status_code = 502
    public_message = "模型服务暂时不可用，请检查后端日志和模型配置"


class EmbeddingConfigError(AppError):
    code = "embedding_key_missing"
    status_code = 503
    public_message = "后端未配置 Embedding 密钥，无法处理知识库"


class EmbeddingUpstreamError(AppError):
    code = "embedding_upstream_error"
    status_code = 502
    public_message = "Embedding 服务调用失败，请检查后端日志"


class QdrantUnavailableError(AppError):
    code = "qdrant_unavailable"
    status_code = 503
    public_message = "向量数据库暂时不可用，请检查 Qdrant 服务和配置"


class CollectionDimensionError(AppError):
    code = "qdrant_dimension_mismatch"
    status_code = 409
    public_message = "Qdrant 集合维度与 EMBEDDING_DIM 不一致，请重建集合"
