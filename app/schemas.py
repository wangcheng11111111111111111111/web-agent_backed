from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: UUID | None = None
    enable_web_search: bool = False
    show_sources: bool = False

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("消息不能为空")
        return value


class ChatResponse(BaseModel):
    answer: str
    session_id: UUID
    memories_used: int = 0


class MessageOut(BaseModel):
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class SessionSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionDetail(SessionSummary):
    messages: list[MessageOut]


class SourceOut(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    content: str


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document: DocumentOut
    indexed_chunks: int


class SearchResponse(BaseModel):
    query: str
    results: list[SourceOut]


class MemoryFactOut(BaseModel):
    id: UUID
    client_id: str
    tenant_id: str
    user_id: str
    fact_type: str
    memory_key: str
    subject: str
    content: str
    confidence: float
    source_session_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


class PromptUpdate(BaseModel):
    prompt: str = Field(..., min_length=1)


class BrandingUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=32)


class BrandingOut(BaseModel):
    name: str
    logo_url: str | None = None
    background_enabled: bool = False
    background_type: Literal["image", "video"] | None = None
    background_url: str | None = None
    updated_at: datetime | None = None


class BackgroundUpdate(BaseModel):
    enabled: bool
    background_type: Literal["image", "video"] | None = None


class BrandingMediaOut(BaseModel):
    kind: Literal["image", "video"]
    filename: str
    url: str
    size_bytes: int
    created_at: datetime
    usage: list[str]


class BrandingMediaUse(BaseModel):
    kind: Literal["image", "video"]
    filename: str
    target: Literal["logo", "background"] = "background"