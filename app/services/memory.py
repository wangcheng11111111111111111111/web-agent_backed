from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx

from app.config import Settings
from app.errors import (
    CollectionDimensionError,
    EmbeddingConfigError,
    QdrantUnavailableError,
)
from app.storage.repositories import MemoryFact, MemoryRepository
from app.services.llm import LLMClient
from app.services.rag import EmbeddingClient

logger = logging.getLogger(__name__)

VALID_FACT_TYPES = {
    "profile",
    "preference",
    "goal",
    "constraint",
    "project",
    "habit",
    "skill",
    "relationship",
    "context",
}
SENSITIVE_PATTERN = re.compile(
    r"(密码|密钥|令牌|验证码|身份证|银行卡|信用卡|"
    r"password|passwd|secret|token|api[\s_-]?key|credit card|otp)",
    re.IGNORECASE,
)
NOISE_PATTERN = re.compile(
    r"(我不知道|我不确定|我无法回答|无法确定|"
    r"not sure|cannot answer|don't know)",
    re.IGNORECASE,
)


@dataclass
class FactCandidate:
    fact_type: str
    memory_key: str
    subject: str
    content: str
    confidence: float
    expires_in_days: int | None


def validate_fact_candidate(
    raw: dict[str, Any],
    minimum_confidence: float,
    minimum_content_chars: int = 4,
    maximum_content_chars: int = 800,
) -> FactCandidate | None:
    fact_type = str(raw.get("fact_type", raw.get("type", ""))).strip().lower()
    content = " ".join(
        str(
            raw.get(
                "value",
                raw.get("content", raw.get("fact", raw.get("text", ""))),
            )
        ).split()
    )
    subject = " ".join(str(raw.get("subject", "用户")).split()) or "用户"
    memory_key = " ".join(
        str(raw.get("key", raw.get("memory_key", ""))).split()
    )
    try:
        confidence = float(raw.get("confidence", 0.65))
    except (TypeError, ValueError):
        return None

    if fact_type not in VALID_FACT_TYPES:
        inferred = f"{memory_key} {content}".casefold()
        if re.search(r"(项目|project|代码库|repository)", inferred):
            fact_type = "project"
        elif re.search(r"(习惯|每天|常常|通常|habit|routine)", inferred):
            fact_type = "habit"
        elif re.search(r"(会|熟练|掌握|技能|skill|熟悉)", inferred):
            fact_type = "skill"
        elif re.search(r"(父亲|母亲|家人|朋友|同事|关系|relationship)", inferred):
            fact_type = "relationship"
        elif re.search(r"(目标|计划|希望|打算|goal|plan)", inferred):
            fact_type = "goal"
        elif re.search(r"(必须|不能|限制|约束|constraint)", inferred):
            fact_type = "constraint"
        elif re.search(r"(喜欢|偏好|更愿意|prefer|favorite)", inferred):
            fact_type = "preference"
        elif re.search(r"(我是|我叫|年龄|职业|居住|profile)", inferred):
            fact_type = "profile"
        else:
            fact_type = "context"
    if fact_type not in VALID_FACT_TYPES:
        return None
    if not minimum_content_chars <= len(content) <= maximum_content_chars:
        return None
    if confidence < minimum_confidence or confidence > 1:
        return None
    if SENSITIVE_PATTERN.search(content) or SENSITIVE_PATTERN.search(subject):
        return None
    if NOISE_PATTERN.search(content):
        return None
    if content.endswith(("?", "？", "吗", "么")):
        return None
    if not memory_key:
        memory_key = f"{fact_type}:{subject}"
    if not re.fullmatch(r"[\w.:-]{2,120}", memory_key, re.UNICODE):
        return None
    try:
        raw_expiry = raw.get("expires_in_days")
        expires_in_days = (
            int(raw_expiry) if raw_expiry not in {None, ""} else None
        )
    except (TypeError, ValueError):
        expires_in_days = None
    if expires_in_days is not None:
        maximum = (
            3650
            if fact_type
            in {"profile", "preference", "habit", "skill", "relationship"}
            else 1095
        )
        if expires_in_days < 1:
            return None
        expires_in_days = min(expires_in_days, maximum)
    else:
        expires_in_days = {
            "goal": 365,
            "constraint": 180,
            "project": 180,
            "context": 90,
        }.get(fact_type)
    return FactCandidate(
        fact_type=fact_type,
        memory_key=memory_key[:120],
        subject=subject[:120],
        content=content,
        confidence=round(confidence, 3),
        expires_in_days=expires_in_days,
    )


class MemoryVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.headers: dict[str, str] = {}
        self._indexes_ready = False
        if settings.qdrant_api_key:
            self.headers["api-key"] = settings.qdrant_api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.qdrant_timeout, connect=5.0),
            headers=self.headers,
        )

    @property
    def collection_url(self) -> str:
        return (
            f"{self.settings.qdrant_url}/collections/"
            f"{self.settings.memory_collection}"
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            return await self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise QdrantUnavailableError() from exc

    async def ensure_collection(self) -> None:
        response = await self._request("GET", self.collection_url)
        if response.status_code == 200:
            try:
                vectors = response.json()["result"]["config"]["params"]["vectors"]
                size = vectors.get("size") if isinstance(vectors, dict) else None
            except (KeyError, TypeError, ValueError):
                size = None
            if size is not None and size != self.settings.embedding_dim:
                raise CollectionDimensionError(
                    "长期记忆集合维度与 EMBEDDING_DIM 不一致"
                )
            await self._ensure_payload_indexes()
            return
        if response.status_code != 404:
            raise QdrantUnavailableError()
        create = await self._request(
            "PUT",
            self.collection_url,
            json={
                "vectors": {
                    "size": self.settings.embedding_dim,
                    "distance": "Cosine",
                }
            },
        )
        if create.status_code not in {200, 201}:
            raise QdrantUnavailableError()
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        if self._indexes_ready:
            return
        for field_name, field_schema in (
            ("client_id", "keyword"),
            ("tenant_id", "keyword"),
            ("user_id", "keyword"),
            ("fact_type", "keyword"),
            ("is_active", "bool"),
        ):
            response = await self._request(
                "PUT",
                f"{self.collection_url}/index?wait=true",
                json={
                    "field_name": field_name,
                    "field_schema": field_schema,
                },
            )
            if response.status_code not in {200, 201, 409}:
                raise QdrantUnavailableError()
        self._indexes_ready = True

    async def upsert(
        self,
        *,
        fact: MemoryFact,
        vector: list[float],
        is_active: bool = True,
    ) -> None:
        await self.ensure_collection()
        response = await self._request(
            "PUT",
            f"{self.collection_url}/points?wait=true",
            json={
                "points": [
                    {
                        "id": str(fact.id),
                        "vector": vector,
                        "payload": {
                            "client_id": fact.client_id,
                            "tenant_id": fact.tenant_id,
                            "user_id": fact.user_id,
                            "fact_id": str(fact.id),
                            "fact_type": fact.fact_type,
                            "memory_key": fact.memory_key,
                            "subject": fact.subject,
                            "content": fact.content,
                            "confidence": fact.confidence,
                            "importance": fact.importance,
                            "updated_at": fact.updated_at.isoformat(),
                            "access_count": fact.access_count,
                            "expires_at": (
                                fact.expires_at.isoformat()
                                if fact.expires_at is not None
                                else None
                            ),
                            "is_active": is_active,
                        },
                    }
                ]
            },
        )
        if response.status_code not in {200, 201}:
            raise QdrantUnavailableError()

    async def search_active(
        self,
        *,
        tenant_id: str,
        user_id: str,
        vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection()
        response = await self._request(
            "POST",
            f"{self.collection_url}/points/search",
            json={
                "vector": vector,
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [
                        {
                            "key": "tenant_id",
                            "match": {"value": tenant_id},
                        },
                        {
                            "key": "user_id",
                            "match": {"value": user_id},
                        }
                    ],
                    "must_not": [
                        {
                            "key": "is_active",
                            "match": {"value": False},
                        }
                    ],
                },
            },
        )
        if response.status_code != 200:
            raise QdrantUnavailableError()
        return response.json().get("result", [])

    async def soft_delete(self, fact_id: UUID) -> None:
        await self.ensure_collection()
        response = await self._request(
            "POST",
            f"{self.collection_url}/points/payload?wait=true",
            json={"points": [str(fact_id)], "payload": {"is_active": False}},
        )
        if response.status_code != 200:
            raise QdrantUnavailableError()

    async def backfill_legacy_scope(self, tenant_id: str) -> int:
        """Add tenant/user payload fields to memories created before scoping."""
        await self.ensure_collection()
        offset: Any = None
        updated = 0
        while True:
            body: dict[str, Any] = {
                "limit": 100,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            response = await self._request(
                "POST",
                f"{self.collection_url}/points/scroll",
                json=body,
            )
            if response.status_code != 200:
                raise QdrantUnavailableError()
            result = response.json().get("result") or {}
            points = result.get("points") or []
            patch_points: list[dict[str, Any]] = []
            for point in points:
                payload = point.get("payload") or {}
                if payload.get("tenant_id") and payload.get("user_id"):
                    continue
                user_id = str(payload.get("user_id") or payload.get("client_id") or "")
                if not user_id:
                    continue
                patch_points.append(
                    {
                        "id": point["id"],
                        "payload": {
                            "tenant_id": str(
                                payload.get("tenant_id") or tenant_id
                            ),
                            "user_id": user_id,
                            "memory_key": str(
                                payload.get("memory_key")
                                or f"{payload.get('fact_type', 'profile')}:"
                                f"{payload.get('subject', '用户')}"
                            ),
                        },
                    }
                )
            if patch_points:
                for item in patch_points:
                    item_response = await self._request(
                        "POST",
                        f"{self.collection_url}/points/payload?wait=true",
                        json={
                            "payload": item["payload"],
                            "points": [item["id"]],
                        },
                    )
                    if item_response.status_code != 200:
                        raise QdrantUnavailableError()
                    updated += 1
            offset = result.get("next_page_offset")
            if offset is None:
                break
        return updated

    async def delete(self, fact_id: UUID) -> None:
        await self.ensure_collection()
        response = await self._request(
            "POST",
            f"{self.collection_url}/points/delete?wait=true",
            json={"points": [str(fact_id)]},
        )
        if response.status_code != 200:
            raise QdrantUnavailableError()


class LongTermMemory:
    def __init__(
        self,
        settings: Settings,
        repository: MemoryRepository,
        embeddings: EmbeddingClient,
        vectors: MemoryVectorStore,
        llm: LLMClient,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.embeddings = embeddings
        self.vectors = vectors
        self.llm = llm

    async def retrieve_safe(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.settings.memory_enabled:
            return [], []
        if not self.settings.embeddings_configured:
            logger.info("Long-term memory retrieval skipped: embedding key missing")
            return [], []
        active: list[dict[str, Any]] = []
        deleted: list[dict[str, Any]] = []
        try:
            [vector] = await self.embeddings.embed([query])
            results = await self.vectors.search_active(
                tenant_id=tenant_id,
                user_id=user_id,
                vector=vector,
                limit=max(self.settings.memory_top_k * 2, self.settings.memory_top_k),
            )
            result_ids: list[UUID] = []
            for item in results:
                payload = item.get("payload") or {}
                raw_id = payload.get("fact_id", item.get("id"))
                try:
                    result_ids.append(UUID(str(raw_id)))
                except (TypeError, ValueError):
                    continue
            active_facts = await self.repository.active_by_ids(
                result_ids,
                tenant_id,
                user_id,
            )
            now = datetime.now(timezone.utc)
            recency_days = max(self.settings.memory_recency_days, 1)
            for item in results:
                similarity = float(item.get("score", 0))
                if similarity < self.settings.memory_retrieval_threshold:
                    continue
                payload = item.get("payload") or {}
                raw_id = payload.get("fact_id", item.get("id"))
                try:
                    fact_id = UUID(str(raw_id))
                except (TypeError, ValueError):
                    continue
                fact = active_facts.get(fact_id)
                if fact is None:
                    continue
                confidence = fact.confidence
                importance = fact.importance
                try:
                    updated = fact.updated_at
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    age_days = max(
                        (now - updated.astimezone(timezone.utc)).total_seconds()
                        / 86400,
                        0,
                    )
                except (TypeError, ValueError):
                    age_days = recency_days
                recency = math.exp(-age_days / recency_days)
                final_score = (
                    similarity * 0.76
                    + importance * 0.18
                    + confidence * 0.04
                    + recency * 0.02
                )
                active.append(
                    {
                        "fact_id": str(fact.id),
                        "fact_type": fact.fact_type,
                        "memory_key": fact.memory_key,
                        "subject": fact.subject,
                        "content": fact.content,
                        "score": round(final_score, 4),
                        "similarity": round(similarity, 4),
                        "importance": round(importance, 4),
                    }
                )
            active.sort(key=lambda memory: memory["score"], reverse=True)
            active = active[: self.settings.memory_top_k]
            touched_ids = [
                UUID(memory["fact_id"])
                for memory in active
                if memory.get("fact_id")
            ]
            try:
                await self.repository.touch(
                    touched_ids,
                    user_id,
                    tenant_id,
                    user_id,
                )
            except Exception as exc:
                logger.warning(
                    "Memory access counter update failed safely (%s)",
                    type(exc).__name__,
                )
        except Exception as exc:
            logger.warning(
                "Long-term memory retrieval failed safely (%s)",
                type(exc).__name__,
            )

        logger.info(
            "Long-term memory retrieval tenant_id=%s user_id=%s "
            "active=%s deleted=%s",
            tenant_id,
            user_id,
            len(active),
            len(deleted),
        )
        return active, deleted

    async def memory_chain(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: UUID,
        user_message: str,
        assistant_answer: str,
    ) -> int:
        if not self.settings.memory_enabled:
            return 0
        if not self.settings.model_configured:
            logger.info("Long-term memory extraction skipped: model config missing")
            return 0
        if not self.settings.embeddings_configured:
            logger.info("Long-term memory extraction skipped: embedding config missing")
            return 0
        try:
            raw_candidates = await self.llm.extract_memories(
                user_message=user_message,
                assistant_answer=assistant_answer,
            )
            validated: list[FactCandidate] = []
            seen: set[str] = set()
            for raw in raw_candidates:
                if not isinstance(raw, dict):
                    continue
                candidate = validate_fact_candidate(
                    raw,
                    self.settings.memory_min_confidence,
                    self.settings.memory_min_content_chars,
                    self.settings.memory_max_content_chars,
                )
                if candidate is None:
                    logger.info("Long-term memory policy rejected a candidate")
                    continue
                key = candidate.content.casefold()
                if key in seen:
                    continue
                seen.add(key)
                validated.append(candidate)
                if len(validated) >= self.settings.memory_max_facts:
                    break

            if not validated:
                logger.info("Long-term memory chain found no valid facts")
                return 0

            vectors = await self.embeddings.embed(
                [candidate.content for candidate in validated]
            )
            stored_count = 0
            explicit_correction = bool(
                re.search(
                    r"(改成|改为|更正|纠正|不再|其实是|更新为|"
                    r"instead|actually|no longer)",
                    user_message,
                    re.IGNORECASE,
                )
            )
            for candidate, vector in zip(validated, vectors, strict=True):
                importance = {
                    "profile": 0.85,
                    "constraint": 0.80,
                    "goal": 0.75,
                    "project": 0.70,
                    "preference": 0.65,
                    "habit": 0.65,
                    "skill": 0.70,
                    "relationship": 0.75,
                    "context": 0.45,
                }.get(candidate.fact_type, 0.5)
                importance = round(
                    importance * 0.7 + candidate.confidence * 0.3,
                    3,
                )
                correction_ids: list[UUID] = []
                existing_id: UUID | None = None
                try:
                    similar = await self.vectors.search_active(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        vector=vector,
                        limit=8,
                    )
                    for item in similar:
                        score = float(item.get("score", 0))
                        payload = item.get("payload") or {}
                        value = payload.get("fact_id")
                        if not value:
                            continue
                        old_id = UUID(str(value))
                        if score >= self.settings.memory_dedup_threshold:
                            existing_id = old_id
                            continue
                        if (
                            explicit_correction
                            and score >= self.settings.memory_correction_threshold
                            and payload.get("fact_type") == candidate.fact_type
                            and payload.get("subject") == candidate.subject
                            and old_id != existing_id
                        ):
                            correction_ids.append(old_id)
                except QdrantUnavailableError:
                    logger.warning(
                        "Memory dedupe unavailable; PostgreSQL hash dedupe remains active"
                    )

                if correction_ids and existing_id is None:
                    logger.info(
                        "Long-term memory correction client_id=%s fact_type=%s "
                        "subject=%s replacing=%s old facts",
                        user_id,
                        candidate.fact_type,
                        candidate.subject,
                        len(correction_ids),
                    )
                    for old_id in correction_ids:
                        try:
                            await self.vectors.soft_delete(old_id)
                        except Exception as exc:
                            logger.warning(
                                "Correction vector soft_delete failed (%s)",
                                type(exc).__name__,
                            )
                            continue
                        await self.repository.soft_delete(
                            fact_id=old_id,
                            client_id=user_id,
                            reason="superseded",
                            tenant_id=tenant_id,
                            user_id=user_id,
                        )

                expires_at = (
                    datetime.now(timezone.utc)
                    + timedelta(days=candidate.expires_in_days)
                    if candidate.expires_in_days is not None
                    else None
                )
                fact = await self.repository.upsert(
                    client_id=user_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    fact_type=candidate.fact_type,
                    memory_key=candidate.memory_key,
                    subject=candidate.subject,
                    content=candidate.content,
                    confidence=candidate.confidence,
                    source_session_id=session_id,
                    fact_id=existing_id or uuid4(),
                    importance=importance,
                    expires_at=expires_at,
                )
                try:
                    await self.vectors.upsert(fact=fact, vector=vector, is_active=True)
                except Exception as exc:
                    logger.warning(
                        "Memory vector write failed after PostgreSQL write (%s)",
                        type(exc).__name__,
                    )
                stored_count += 1
            logger.info(
                "Long-term memory chain completed tenant_id=%s user_id=%s stored=%s",
                tenant_id,
                user_id,
                stored_count,
            )
            return stored_count
        except Exception as exc:
            logger.warning(
                "Long-term memory chain failed safely (%s: %s)",
                type(exc).__name__,
                str(exc),
            )
            return 0

    async def delete(
        self,
        tenant_id: str,
        user_id: str,
        fact_id: UUID,
    ) -> bool:
        deleted = await self.repository.soft_delete(
            fact_id=fact_id,
            client_id=user_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if not deleted:
            return False
        try:
            await self.vectors.soft_delete(fact_id)
        except Exception as exc:
            logger.warning(
                "Memory vector cleanup deferred; PostgreSQL deletion is authoritative (%s)",
                type(exc).__name__,
            )
        return True


def format_memory_context(
    active_memories: list[dict[str, Any]],
    deleted_memories: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []
    for memory in active_memories:
        parts.append(
            f"[类型: {memory.get('fact_type', 'profile')} | "
            f"Key: {memory.get('memory_key', '-')} | "
            f"主体: {memory.get('subject', '用户')}]\n"
            f"{memory.get('content', '')}"
        )
    section = ""
    if parts:
        section += "以下是关于该用户的已知事实，请在回答时参考：\n\n"
        section += "\n\n".join(parts)
    if deleted_memories:
        deleted_parts = []
        for memory in deleted_memories:
            deleted_parts.append(
                f"[类型: {memory.get('fact_type', 'profile')} | "
                f"主体: {memory.get('subject', '用户')}]\n"
                f"{memory.get('content', '')}"
            )
        section += "\n\n⚠️ 以下是用户明确删除或纠正过的事实，回答时**禁止引用**这些内容，如与用户当前消息冲突请以用户最新说法为准：\n\n"
        section += "\n\n".join(deleted_parts)
    return section
