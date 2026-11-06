from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from app.database import Database
from app.errors import DatabaseUnavailableError, SessionNotFoundError
from app.services.short_term_memory import ShortTermMemory


@dataclass
class ChatStart:
    session_id: UUID
    history: list[dict[str, str]]
    user_message_id: UUID
    client_id: str


@dataclass
class StoredDocument:
    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    created_at: datetime


@dataclass
class MemoryFact:
    id: UUID
    client_id: str
    tenant_id: str
    user_id: str
    fact_type: str
    memory_key: str
    subject: str
    content: str
    confidence: float
    source_session_id: UUID | None
    created_at: datetime
    updated_at: datetime
    content_hash: str = ""
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None
    deleted_reason: str | None = None


class ChatRepository:
    def __init__(
        self,
        database: Database,
        short_term_memory: ShortTermMemory | None = None,
    ) -> None:
        self.database = database
        self.short_term_memory = short_term_memory

    @staticmethod
    def _title(message: str) -> str:
        single_line = " ".join(message.split())
        return single_line[:48] + ("…" if len(single_line) > 48 else "")

    async def begin_chat(
        self,
        message: str,
        session_id: UUID | None,
        client_id: str = "default",
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> ChatStart:
        user_id = user_id or client_id
        user_message_id = uuid4()
        title = self._title(message)
        cached_history: list[dict[str, Any]] | None = None
        cache_hit = False
        history: list[dict[str, str]] = []
        try:
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    if session_id is None:
                        session_id = uuid4()
                        await connection.execute(
                            """
                            INSERT INTO sessions (
                                id, title, client_id, tenant_id, user_id
                            )
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            session_id,
                            title,
                            client_id,
                            tenant_id,
                            user_id,
                        )
                    else:
                        owned = await connection.fetchrow(
                            """
                            SELECT title
                            FROM sessions
                            WHERE id = $1
                              AND client_id = $2
                              AND tenant_id = $3
                              AND user_id = $4
                            """,
                            session_id,
                            client_id,
                            tenant_id,
                            user_id,
                        )
                        if owned is None:
                            raise SessionNotFoundError()
                        title = owned["title"]

                    if self.short_term_memory is not None:
                        cached_history = (
                            await self.short_term_memory.get_history(session_id)
                        )
                        cache_hit = cached_history is not None
                    if cached_history is None:
                        records = await connection.fetch(
                            """
                            SELECT id, role, content, created_at
                            FROM messages
                            WHERE session_id = $1
                              AND role IN ('user', 'assistant')
                            ORDER BY seq DESC
                            LIMIT $2
                            """,
                            session_id,
                            (
                                self.short_term_memory.settings.short_term_message_limit
                                if self.short_term_memory is not None
                                else 40
                            ),
                        )
                        cached_history = [
                            {
                                "id": str(record["id"]),
                                "role": record["role"],
                                "content": record["content"],
                                "created_at": record["created_at"].isoformat(),
                            }
                            for record in reversed(records)
                        ]
                    history = [
                        {"role": item["role"], "content": item["content"]}
                        for item in cached_history
                        if item.get("role") in {"user", "assistant"}
                    ]
                    user_created_at = await connection.fetchval(
                        """
                        INSERT INTO messages (id, session_id, role, content)
                        VALUES ($1, $2, 'user', $3)
                        RETURNING created_at
                        """,
                        user_message_id,
                        session_id,
                        message,
                    )
                    await connection.execute(
                        "UPDATE sessions SET updated_at = NOW() WHERE id = $1",
                        session_id,
                    )
        except SessionNotFoundError:
            raise
        except Exception as exc:
            if isinstance(exc, DatabaseUnavailableError):
                raise
            raise DatabaseUnavailableError() from exc

        if self.short_term_memory is not None:
            if not cache_hit:
                await self.short_term_memory.cache_history(
                    session_id,
                    cached_history or [],
                    title=title,
                )
            await self.short_term_memory.append_message(
                session_id,
                {
                    "id": str(user_message_id),
                    "role": "user",
                    "content": message,
                    "created_at": user_created_at.isoformat(),
                },
                create_if_missing=True,
            )
        return ChatStart(
            session_id=session_id,
            history=history,
            user_message_id=user_message_id,
            client_id=client_id,
        )

    async def finish_chat(self, session_id: UUID, answer: str) -> UUID:
        assistant_id = uuid4()
        assistant_created_at: datetime | None = None
        try:
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    assistant_created_at = await connection.fetchval(
                        """
                        INSERT INTO messages (id, session_id, role, content)
                        VALUES ($1, $2, 'assistant', $3)
                        RETURNING created_at
                        """,
                        assistant_id,
                        session_id,
                        answer,
                    )
                    await connection.execute(
                        "UPDATE sessions SET updated_at = NOW() WHERE id = $1",
                        session_id,
                    )
        except Exception as exc:
            if isinstance(exc, DatabaseUnavailableError):
                raise
            raise DatabaseUnavailableError() from exc
        if self.short_term_memory is not None and assistant_created_at is not None:
            await self.short_term_memory.append_message(
                session_id,
                {
                    "id": str(assistant_id),
                    "role": "assistant",
                    "content": answer,
                    "created_at": assistant_created_at.isoformat(),
                },
            )
        return assistant_id

    async def list_sessions(
        self,
        client_id: str = "default",
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            records = await connection.fetch(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id)::int AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.client_id = $1
                  AND s.tenant_id = $2
                  AND s.user_id = $3
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """,
                client_id,
                tenant_id,
                user_id,
            )
        return [dict(record) for record in records]

    async def get_session(
        self,
        session_id: UUID,
        client_id: str = "default",
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            session = await connection.fetchrow(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                WHERE id = $1
                  AND client_id = $2
                  AND tenant_id = $3
                  AND user_id = $4
                """,
                session_id,
                client_id,
                tenant_id,
                user_id,
            )
            if session is None:
                return None
            messages = await connection.fetch(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE session_id = $1
                ORDER BY seq ASC
                """,
                session_id,
            )
        result = dict(session)
        result["message_count"] = len(messages)
        result["messages"] = [dict(message) for message in messages]
        if self.short_term_memory is not None:
            await self.short_term_memory.cache_history(
                session_id,
                [
                    {
                        "id": str(message["id"]),
                        "role": message["role"],
                        "content": message["content"],
                        "created_at": message["created_at"].isoformat(),
                    }
                    for message in messages
                ],
                title=str(result["title"]),
            )
        return result

    async def get_message_content(
        self,
        message_id: UUID,
        *,
        client_id: str,
        tenant_id: str,
        user_id: str,
        role: str = "assistant",
    ) -> str | None:
        if self.short_term_memory is not None and role == "assistant":
            cached = await self.short_term_memory.get_message_content(message_id)
            if cached is not None:
                return cached
        async with self.database.acquire() as connection:
            content = await connection.fetchval(
                """
                SELECT m.content
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE m.id = $1
                  AND m.role = $2
                  AND s.client_id = $3
                  AND s.tenant_id = $4
                  AND s.user_id = $5
                """,
                message_id,
                role,
                client_id,
                tenant_id,
                user_id,
            )
        if content is None:
            return None
        text = str(content)
        if self.short_term_memory is not None and role == "assistant":
            await self.short_term_memory.cache_message_content(message_id, text)
        return text

    async def delete_session(
        self,
        session_id: UUID,
        client_id: str = "default",
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> bool:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            result = await connection.execute(
                """
                DELETE FROM sessions
                WHERE id = $1
                  AND client_id = $2
                  AND tenant_id = $3
                  AND user_id = $4
                """,
                session_id,
                client_id,
                tenant_id,
                user_id,
            )
        if result == "DELETE 1" and self.short_term_memory is not None:
            await self.short_term_memory.delete_session(session_id)
        return result == "DELETE 1"

    async def recall_recent_messages(
        self,
        *,
        client_id: str,
        tenant_id: str,
        user_id: str,
        exclude_session_id: UUID,
        limit: int,
    ) -> list[dict[str, Any]]:
        async with self.database.acquire() as connection:
            records = await connection.fetch(
                """
                SELECT m.id, m.session_id, m.role, m.content, m.seq,
                       m.created_at, s.title AS session_title
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.client_id = $1
                  AND s.tenant_id = $2
                  AND s.user_id = $3
                  AND m.session_id <> $4
                ORDER BY m.seq DESC
                LIMIT $5
                """,
                client_id,
                tenant_id,
                user_id,
                exclude_session_id,
                limit,
            )
        return [dict(record) for record in reversed(records)]


class MemoryRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def content_hash(content: str) -> str:
        normalized = re.sub(r"\s+", " ", content.strip()).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def upsert(
        self,
        *,
        client_id: str,
        tenant_id: str = "default",
        user_id: str | None = None,
        fact_type: str,
        memory_key: str | None = None,
        subject: str,
        content: str,
        confidence: float,
        source_session_id: UUID | None,
        fact_id: UUID | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> MemoryFact:
        user_id = user_id or client_id
        memory_key = (memory_key or f"{fact_type}:{subject}")[:120]
        fact_id = fact_id or uuid4()
        content_hash = self.content_hash(content)
        async with self.database.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"{tenant_id}:{user_id}:{content_hash}",
                )
                exact = await connection.fetchrow(
                    """
                    SELECT id
                    FROM user_facts
                    WHERE tenant_id = $1
                      AND user_id = $2
                      AND content_hash = $3
                      AND is_active = TRUE
                    FOR UPDATE
                    """,
                    tenant_id,
                    user_id,
                    content_hash,
                )
                if exact is not None:
                    fact_id = exact["id"]
                record = await connection.fetchrow(
                    """
                    INSERT INTO user_facts (
                        id, client_id, fact_type, subject, content,
                        confidence, source_session_id, content_hash,
                        importance, is_active, deleted_reason, tenant_id,
                        user_id, memory_key, expires_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9,
                        TRUE, NULL, $10, $11, $12, $13
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        fact_type = EXCLUDED.fact_type,
                        subject = EXCLUDED.subject,
                        content = EXCLUDED.content,
                        confidence = EXCLUDED.confidence,
                        source_session_id = EXCLUDED.source_session_id,
                        content_hash = EXCLUDED.content_hash,
                        importance = EXCLUDED.importance,
                        tenant_id = EXCLUDED.tenant_id,
                        user_id = EXCLUDED.user_id,
                        memory_key = EXCLUDED.memory_key,
                        expires_at = EXCLUDED.expires_at,
                        is_active = TRUE,
                        deleted_reason = NULL,
                        updated_at = NOW()
                    WHERE user_facts.client_id = EXCLUDED.client_id
                      AND user_facts.tenant_id = EXCLUDED.tenant_id
                      AND user_facts.user_id = EXCLUDED.user_id
                    RETURNING id, client_id, fact_type, subject, content,
                              confidence, source_session_id, created_at, updated_at,
                              content_hash, importance, access_count,
                              last_accessed_at, expires_at, deleted_reason,
                              tenant_id, user_id, memory_key
                    """,
                    fact_id,
                    client_id,
                    fact_type,
                    subject,
                    content,
                    confidence,
                    source_session_id,
                    content_hash,
                    importance,
                    tenant_id,
                    user_id,
                    memory_key,
                    expires_at,
                )
        if record is None:
            raise ValueError("记忆写入冲突")
        return MemoryFact(**dict(record))

    async def list_active(
        self,
        client_id: str,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> list[MemoryFact]:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            await connection.execute(
                """
                UPDATE user_facts
                SET is_active = FALSE,
                    deleted_reason = 'expired',
                    updated_at = NOW()
                WHERE tenant_id = $1
                  AND user_id = $2
                  AND is_active = TRUE
                  AND expires_at IS NOT NULL
                  AND expires_at <= NOW()
                """,
                tenant_id,
                user_id,
            )
            records = await connection.fetch(
                """
                SELECT id, client_id, fact_type, subject, content, confidence,
                       source_session_id, created_at, updated_at, content_hash,
                       importance, access_count, last_accessed_at, expires_at,
                       deleted_reason, tenant_id, user_id, memory_key
                FROM user_facts
                WHERE tenant_id = $1
                  AND user_id = $2
                  AND is_active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY updated_at DESC
                """,
                tenant_id,
                user_id,
            )
        return [MemoryFact(**dict(record)) for record in records]

    async def list(
        self,
        client_id: str,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> list[MemoryFact]:
        """Compatibility alias used by the memory API."""
        return await self.list_active(client_id, tenant_id, user_id)

    async def find_active_by_subject(
        self,
        client_id: str,
        fact_type: str,
        subject: str,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> list[MemoryFact]:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            records = await connection.fetch(
                """
                SELECT id, client_id, fact_type, subject, content, confidence,
                       source_session_id, created_at, updated_at, content_hash,
                       importance, access_count, last_accessed_at, expires_at,
                       deleted_reason, tenant_id, user_id, memory_key
                FROM user_facts
                WHERE tenant_id = $1
                  AND user_id = $2
                  AND is_active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND fact_type = $3
                  AND subject = $4
                ORDER BY confidence DESC
                """,
                tenant_id,
                user_id,
                fact_type,
                subject,
            )
        return [MemoryFact(**dict(record)) for record in records]

    async def list_recent_deleted(
        self,
        client_id: str,
        limit: int = 5,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> list[MemoryFact]:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            records = await connection.fetch(
                """
                SELECT id, client_id, fact_type, subject, content, confidence,
                       source_session_id, created_at, updated_at, content_hash,
                       importance, access_count, last_accessed_at, expires_at,
                       deleted_reason, tenant_id, user_id, memory_key
                FROM user_facts
                WHERE tenant_id = $1
                  AND user_id = $2
                  AND is_active = FALSE
                ORDER BY updated_at DESC
                LIMIT $3
                """,
                tenant_id,
                user_id,
                limit,
            )
        return [MemoryFact(**dict(record)) for record in records]

    async def touch(
        self,
        fact_ids: list[UUID],
        client_id: str,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> None:
        if not fact_ids:
            return
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            await connection.execute(
                """
                UPDATE user_facts
                SET access_count = access_count + 1,
                    last_accessed_at = NOW()
                WHERE tenant_id = $1
                  AND user_id = $2
                  AND id = ANY($3::uuid[])
                  AND is_active = TRUE
                """,
                tenant_id,
                user_id,
                fact_ids,
            )

    async def soft_delete(
        self,
        fact_id: UUID,
        client_id: str,
        reason: str = "user_deleted",
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> bool:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            result = await connection.execute(
                """
                UPDATE user_facts
                SET is_active = FALSE,
                    deleted_reason = $5,
                    updated_at = NOW()
                WHERE id = $1
                  AND client_id = $2
                  AND tenant_id = $3
                  AND user_id = $4
                  AND is_active = TRUE
                """,
                fact_id,
                client_id,
                tenant_id,
                user_id,
                reason,
            )
        return result == "UPDATE 1"

    async def soft_delete_by_subject(
        self,
        client_id: str,
        fact_type: str,
        subject: str,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> int:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            result = await connection.execute(
                """
                UPDATE user_facts
                SET is_active = FALSE,
                    deleted_reason = 'superseded',
                    updated_at = NOW()
                WHERE tenant_id = $1
                  AND user_id = $2
                  AND fact_type = $3
                  AND subject = $4
                  AND is_active = TRUE
                """,
                tenant_id,
                user_id,
                fact_type,
                subject,
            )
        if result.startswith("UPDATE "):
            try:
                return int(result.split(" ")[1])
            except (IndexError, ValueError):
                return 0
        return 0

    async def hard_delete(
        self,
        fact_id: UUID,
        client_id: str,
        tenant_id: str = "default",
        user_id: str | None = None,
    ) -> bool:
        user_id = user_id or client_id
        async with self.database.acquire() as connection:
            result = await connection.execute(
                """
                DELETE FROM user_facts
                WHERE id = $1
                  AND client_id = $2
                  AND tenant_id = $3
                  AND user_id = $4
                """,
                fact_id,
                client_id,
                tenant_id,
                user_id,
            )
        return result == "DELETE 1"


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def create(
        self,
        *,
        document_id: UUID | None = None,
        filename: str,
        content_type: str,
        size_bytes: int,
        chunks: list[dict[str, Any]],
    ) -> StoredDocument:
        document_id = document_id or uuid4()
        try:
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    record = await connection.fetchrow(
                        """
                        INSERT INTO documents
                            (id, filename, content_type, size_bytes, chunk_count)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id, filename, content_type, size_bytes,
                                  chunk_count, created_at
                        """,
                        document_id,
                        filename,
                        content_type,
                        size_bytes,
                        len(chunks),
                    )
                    for chunk in chunks:
                        await connection.execute(
                            """
                            INSERT INTO document_chunks
                                (id, document_id, chunk_index, content, source)
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            chunk["id"],
                            document_id,
                            chunk["chunk_index"],
                            chunk["content"],
                            chunk["source"],
                        )
        except Exception as exc:
            if isinstance(exc, DatabaseUnavailableError):
                raise
            raise DatabaseUnavailableError() from exc
        assert record is not None
        return StoredDocument(**dict(record))

    async def list(self) -> list[StoredDocument]:
        async with self.database.acquire() as connection:
            records = await connection.fetch(
                """
                SELECT id, filename, content_type, size_bytes,
                       chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC
                """
            )
        return [StoredDocument(**dict(record)) for record in records]

    async def active_ids(self, document_ids: list[UUID]) -> set[str]:
        if not document_ids:
            return set()
        async with self.database.acquire() as connection:
            records = await connection.fetch(
                """
                SELECT id
                FROM documents
                WHERE id = ANY($1::uuid[])
                """,
                document_ids,
            )
        return {str(record["id"]) for record in records}

    async def delete(self, document_id: UUID) -> bool:
        async with self.database.acquire() as connection:
            result = await connection.execute(
                "DELETE FROM documents WHERE id = $1",
                document_id,
            )
        return result == "DELETE 1"
