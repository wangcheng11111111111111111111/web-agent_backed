from __future__ import annotations

import asyncio
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import httpx
from pypdf import PdfReader

from app.config import Settings
from app.errors import (
    CollectionDimensionError,
    EmbeddingConfigError,
    EmbeddingUpstreamError,
    QdrantUnavailableError,
)
from app.storage.repositories import DocumentRepository, StoredDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class ParsedChunk:
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    source: str
    filename: str


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.embeddings_configured:
            raise EmbeddingConfigError()
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.embedding_base_url}/embeddings"
        vectors: list[list[float]] = []
        timeout = httpx.Timeout(self.settings.embedding_timeout, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                for offset in range(0, len(texts), 16):
                    batch = texts[offset : offset + 16]
                    response = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": self.settings.embedding_model,
                            "input": batch,
                        },
                    )
                    if response.status_code >= 400:
                        logger.error(
                            "Embedding upstream returned HTTP %s",
                            response.status_code,
                        )
                        raise EmbeddingUpstreamError()
                    payload = response.json()
                    rows = sorted(
                        payload.get("data", []),
                        key=lambda row: row.get("index", 0),
                    )
                    if len(rows) != len(batch):
                        raise EmbeddingUpstreamError(
                            "Embedding 服务返回的向量数量不完整"
                        )
                    for row in rows:
                        vector = row.get("embedding")
                        if not isinstance(vector, list):
                            raise EmbeddingUpstreamError(
                                "Embedding 服务返回了无效向量"
                            )
                        if len(vector) != self.settings.embedding_dim:
                            raise EmbeddingUpstreamError(
                                "Embedding 向量维度与 EMBEDDING_DIM 不一致"
                            )
                        vectors.append(vector)
        except (EmbeddingConfigError, EmbeddingUpstreamError):
            raise
        except (httpx.HTTPError, ValueError) as exc:
            logger.error(
                "Embedding request failed: %s", type(exc).__name__
            )
            raise EmbeddingUpstreamError() from exc
        return vectors


class QdrantStore:
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
            f"{self.settings.qdrant_collection}"
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
            logger.error("Qdrant request failed: %s", type(exc).__name__)
            raise QdrantUnavailableError() from exc

    async def ensure_collection(self) -> None:
        response = await self._request("GET", self.collection_url)
        if response.status_code == 200:
            try:
                config = response.json()["result"]["config"]["params"]["vectors"]
                actual_size = (
                    config["size"]
                    if isinstance(config, dict) and "size" in config
                    else None
                )
            except (KeyError, TypeError, ValueError):
                actual_size = None
            if actual_size is not None and actual_size != self.settings.embedding_dim:
                raise CollectionDimensionError(
                    "Qdrant 集合维度为 "
                    f"{actual_size}，配置要求 {self.settings.embedding_dim}。"
                    "请删除旧集合后重新上传文档。"
                )
            await self._ensure_payload_indexes()
            return
        if response.status_code != 404:
            logger.error(
                "Qdrant collection check returned HTTP %s",
                response.status_code,
            )
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
            logger.error(
                "Qdrant collection creation returned HTTP %s",
                create.status_code,
            )
            raise QdrantUnavailableError()
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        if self._indexes_ready:
            return
        for field_name in ("document_id", "filename"):
            response = await self._request(
                "PUT",
                f"{self.collection_url}/index?wait=true",
                json={
                    "field_name": field_name,
                    "field_schema": "keyword",
                },
            )
            if response.status_code not in {200, 201, 409}:
                raise QdrantUnavailableError()
        self._indexes_ready = True

    async def upsert(
        self, chunks: list[ParsedChunk], vectors: list[list[float]]
    ) -> None:
        await self.ensure_collection()
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                {
                    "id": str(chunk.id),
                    "vector": vector,
                    "payload": {
                        "document_id": str(chunk.document_id),
                        "filename": chunk.filename,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.source,
                        "text": chunk.content,
                    },
                }
            )
        response = await self._request(
            "PUT",
            f"{self.collection_url}/points?wait=true",
            json={"points": points},
        )
        if response.status_code not in {200, 201}:
            logger.error(
                "Qdrant upsert returned HTTP %s", response.status_code
            )
            raise QdrantUnavailableError()

    async def search(
        self, vector: list[float], limit: int
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
            },
        )
        if response.status_code != 200:
            logger.error(
                "Qdrant search returned HTTP %s", response.status_code
            )
            raise QdrantUnavailableError()
        try:
            return response.json().get("result", [])
        except ValueError as exc:
            raise QdrantUnavailableError() from exc

    async def delete_document(self, document_id: UUID) -> None:
        await self.ensure_collection()
        response = await self._request(
            "POST",
            f"{self.collection_url}/points/delete?wait=true",
            json={
                "filter": {
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"value": str(document_id)},
                        }
                    ]
                }
            },
        )
        if response.status_code != 200:
            logger.error(
                "Qdrant document deletion returned HTTP %s",
                response.status_code,
            )
            raise QdrantUnavailableError()


class RagService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingClient,
        qdrant: QdrantStore,
        document_repository: DocumentRepository | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.document_repository = document_repository

    async def index_chunks(self, chunks: list[ParsedChunk]) -> None:
        texts = [chunk.content for chunk in chunks]
        vectors = await self.embeddings.embed(texts)
        batch_size = 100
        for offset in range(0, len(chunks), batch_size):
            batch_chunks = chunks[offset : offset + batch_size]
            batch_vectors = vectors[offset : offset + batch_size]
            await self.qdrant.upsert(batch_chunks, batch_vectors)

    async def search(self, query: str) -> list[dict[str, Any]]:
        [vector] = await self.embeddings.embed([query])
        raw_results = await self.qdrant.search(
            vector, self.settings.rag_top_k
        )
        active_document_ids: set[str] | None = None
        if self.document_repository is not None:
            candidate_ids: list[UUID] = []
            for item in raw_results:
                value = (item.get("payload") or {}).get("document_id")
                if not value:
                    continue
                try:
                    candidate_ids.append(UUID(str(value)))
                except ValueError:
                    continue
            active_document_ids = await self.document_repository.active_ids(
                list(set(candidate_ids))
            )
        results: list[dict[str, Any]] = []
        for item in raw_results:
            score = float(item.get("score", 0.0))
            if score < self.settings.rag_score_threshold:
                continue
            payload = item.get("payload") or {}
            document_id = str(payload.get("document_id", ""))
            if (
                active_document_ids is not None
                and document_id not in active_document_ids
            ):
                continue
            results.append(
                {
                    "document_id": document_id,
                    "filename": str(payload.get("filename", "未知文档")),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "score": round(score, 4),
                    "content": str(payload.get("text", "")),
                }
            )
        return results

    async def search_safe(self, query: str) -> list[dict[str, Any]]:
        if not self.settings.embeddings_configured:
            logger.info("RAG skipped: embedding config is incomplete")
            return []
        try:
            return await self.search(query)
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed; continuing normal chat (%s: %s)",
                type(exc).__name__,
                str(exc),
            )
            return []


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            [paragraph]
            if len(paragraph) <= chunk_size
            else [
                paragraph[i : i + chunk_size]
                for i in range(0, len(paragraph), chunk_size - overlap)
            ]
        )
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{piece}".strip()
            else:
                current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def parse_document(
    filename: str,
    content: bytes,
    settings: Settings,
    document_id: UUID,
) -> list[ParsedChunk]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("仅支持 Markdown、TXT 和 PDF 文档")

    if extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError("PDF 解析失败，请确认文件完整且未加密") from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("gb18030", errors="replace")

    text_chunks = split_text(
        text,
        settings.rag_chunk_size,
        settings.rag_chunk_overlap,
    )
    if not text_chunks:
        raise ValueError("文档中没有可索引的文本")

    safe_name = Path(filename).name
    chunks: list[ParsedChunk] = []
    for index, chunk in enumerate(text_chunks):
        chunks.append(
            ParsedChunk(
                id=uuid5(NAMESPACE_URL, f"document:{document_id}:chunk:{index}"),
                document_id=document_id,
                chunk_index=index,
                content=chunk,
                source=f"{safe_name}#chunk-{index + 1}",
                filename=safe_name,
            )
        )
    return chunks


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        repository: DocumentRepository,
        rag: RagService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.rag = rag

    async def upload(
        self,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> tuple[StoredDocument, int]:
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError(
                f"文件不能超过 {self.settings.max_upload_mb} MB"
            )
        safe_name = Path(filename).name
        if not safe_name:
            raise ValueError("文件名无效")

        document_id = uuid4()
        chunks = await asyncio.to_thread(
            parse_document,
            safe_name,
            content,
            self.settings,
            document_id,
        )
        stored = await self.repository.create(
            document_id=document_id,
            filename=safe_name,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(content),
            chunks=[
                {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "source": chunk.source,
                }
                for chunk in chunks
            ],
        )
        try:
            await self.rag.index_chunks(chunks)
        except Exception:
            logger.exception(
                "Document indexing failed; removing metadata document_id=%s",
                stored.id,
            )
            await self.repository.delete(stored.id)
            raise
        logger.info(
            "Document indexed document_id=%s chunks=%s",
            stored.id,
            len(chunks),
        )
        return stored, len(chunks)

    async def delete(self, document_id: UUID) -> bool:
        await self.rag.qdrant.delete_document(document_id)
        return await self.repository.delete(document_id)
