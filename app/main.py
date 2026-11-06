from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sys
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlencode
from uuid import UUID

# ONNX Runtime emits provider fallback diagnostics directly from its native
# library, bypassing Python's warnings module. Keep fatal errors only because
# CosyVoice can safely use its CPU fallback when CUDA provider DLLs are absent.
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
os.environ.setdefault("COSYVOICE_ONNX_PROVIDER", "cpu")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*LoRACompatibleLinear.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.nn\.utils\.weight_norm.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Specified provider 'CUDAExecutionProvider'.*",
)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# PyCharm and `python app/main.py` execute this file as a script. Ensure the
# backend package root is importable and prefer the project's pinned packages
# without replacing the process, which IDEs treat as an immediate exit.
if __package__ in {None, ""}:
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root))
    venv_site_packages = (
        backend_root / ".venv" / "Lib" / "site-packages"
        if os.name == "nt"
        else backend_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    if venv_site_packages.is_dir():
        sys.path.insert(0, str(venv_site_packages))

from app.agent.graph import AssistantGraph
from app.storage.client_branding_store import BrandingStore
from app.config import DEFAULT_PERSONALITY_PROMPT, Settings, get_settings
from app.database import Database
from app.errors import AppError
from app.storage.media_storage import MediaStorage
from app.storage.prompt_store import PromptStore
from app.storage.repositories import (
    ChatRepository,
    DocumentRepository,
    MemoryRepository,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    BrandingOut,
    BrandingUpdate,
    BackgroundUpdate,
    BrandingMediaOut,
    BrandingMediaUse,
    DocumentOut,
    DocumentUploadResponse,
    MemoryFactOut,
    PromptUpdate,
    SearchResponse,
    SessionDetail,
    SessionSummary,
)
from app.services.audio import TTSService
from app.services.llm import LLMClient
from app.services.conversation_memory import ConversationRecall
from app.services.memory import (
    LongTermMemory,
    MemoryVectorStore,
)
from app.services.rag import (
    DocumentService,
    EmbeddingClient,
    QdrantStore,
    RagService,
)
from app.services.short_term_memory import ShortTermMemory
from app.services.web_search import WebSearchService
from app.services.crawler import WebCrawlerService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    database: Database
    chat_repository: ChatRepository
    document_repository: DocumentRepository
    memory_repository: MemoryRepository
    agent: AssistantGraph
    rag: RagService
    documents: DocumentService
    web_search: WebSearchService
    web_crawler: WebCrawlerService
    short_term_memory: ShortTermMemory
    conversation_recall: ConversationRecall
    long_term_memory: LongTermMemory
    prompt_store: PromptStore
    branding_store: BrandingStore
    media_storage: MediaStorage
    tts: TTSService
    memory_tasks: set[asyncio.Task[Any]] = field(default_factory=set)


def _build_container(settings: Settings) -> Container:
    database = Database(settings)
    short_term_memory = ShortTermMemory(settings)
    chat_repository = ChatRepository(database, short_term_memory)
    conversation_recall = ConversationRecall(settings, chat_repository)
    document_repository = DocumentRepository(database)
    memory_repository = MemoryRepository(database)
    prompt_store = PromptStore(database)
    branding_store = BrandingStore(database)
    media_storage = MediaStorage(settings)
    llm = LLMClient(settings, prompt_store=prompt_store)
    embeddings = EmbeddingClient(settings)
    qdrant = QdrantStore(settings)
    rag = RagService(settings, embeddings, qdrant, document_repository)
    memory_vectors = MemoryVectorStore(settings)
    web_search = WebSearchService(settings)
    web_crawler = WebCrawlerService(settings)
    long_term_memory = LongTermMemory(
        settings,
        memory_repository,
        embeddings,
        memory_vectors,
        llm,
    )
    agent = AssistantGraph(
        llm,
        rag,
        long_term_memory=long_term_memory,
        web_search=web_search,
        conversation_recall=conversation_recall,
        web_crawler=web_crawler,
        document_repository=document_repository,
        memory_repository=memory_repository,
    )
    documents = DocumentService(settings, document_repository, rag)
    tts = TTSService(settings)
    return Container(
        settings=settings,
        database=database,
        chat_repository=chat_repository,
        document_repository=document_repository,
        memory_repository=memory_repository,
        agent=agent,
        rag=rag,
        documents=documents,
        web_search=web_search,
        web_crawler=web_crawler,
        short_term_memory=short_term_memory,
        conversation_recall=conversation_recall,
        long_term_memory=long_term_memory,
        prompt_store=prompt_store,
        branding_store=branding_store,
        media_storage=media_storage,
        tts=tts,
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _document_out(document: Any) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )


def _branding_out(branding: Any, client_id: str) -> BrandingOut:
    version = (
        str(int(branding.updated_at.timestamp() * 1_000_000))
        if branding.updated_at is not None
        else "default"
    )

    def asset_url(path: str) -> str:
        return f"{path}?{urlencode({'client_id': client_id, 'v': version})}"

    background_url: str | None = None
    if branding.background_enabled:
        if branding.background_type == "video" and branding.background_video_file:
            background_url = asset_url(
                "/api/settings/branding/background/video"
            )
        elif branding.background_type == "image" and branding.logo_file:
            background_url = asset_url("/api/settings/branding/logo")
    return BrandingOut(
        name=branding.name,
        logo_url=(
            asset_url("/api/settings/branding/logo")
            if branding.logo_file
            else None
        ),
        background_enabled=branding.background_enabled,
        background_type=branding.background_type,
        background_url=background_url,
        updated_at=branding.updated_at,
    )


CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _client_id(request: Request, *, allow_query: bool = False) -> str:
    value = request.app.state.container.settings.global_client_id
    if not CLIENT_ID_PATTERN.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_client_id",
                "message": "GLOBAL_CLIENT_ID 必须是 8-64 位字母、数字、下划线或连字符",
            },
        )
    return value


def _identity(request: Request) -> tuple[str, str, str]:
    client_id = _client_id(request)
    settings = request.app.state.container.settings
    tenant_id = settings.global_tenant_id
    user_id = settings.global_user_id
    for name, value in (
        ("GLOBAL_TENANT_ID", tenant_id),
        ("GLOBAL_USER_ID", user_id),
    ):
        if not CLIENT_ID_PATTERN.fullmatch(value):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_identity",
                    "message": f"{name} 必须是 8-64 位字母、数字、下划线或连字符",
                },
            )
    return client_id, tenant_id, user_id


def _schedule_memory(
    container: Container,
    *,
    session_id: UUID,
    client_id: str,
    tenant_id: str,
    user_id: str,
    user_message: str,
    assistant_answer: str,
) -> None:
    task = asyncio.create_task(
        container.long_term_memory.memory_chain(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_answer=assistant_answer,
        )
    )
    container.memory_tasks.add(task)

    def finish(done: asyncio.Task[Any]) -> None:
        container.memory_tasks.discard(done)
        if done.cancelled():
            return
        try:
            done.result()
        except Exception:
            logger.exception("Background memory chain failed")

    task.add_done_callback(finish)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    container = _build_container(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Runtime targets PostgreSQL=%s:%s/%s Redis=%s:%s",
            resolved_settings.pg_host,
            resolved_settings.pg_port,
            resolved_settings.pg_database,
            resolved_settings.redis_host,
            resolved_settings.redis_port,
        )
        try:
            await container.short_term_memory.connect()
        except Exception:
            logger.warning("Redis short-term memory unavailable")

        if container.tts.configured:
            await container.tts.warmup()

        yield
        for task in list(container.memory_tasks):
            task.cancel()
        if container.memory_tasks:
            await asyncio.gather(
                *container.memory_tasks,
                return_exceptions=True,
            )
        await container.short_term_memory.close()
        await container.database.close()

    app = FastAPI(
        title="Web AI Assistant API",
        version="1.0.0",
        description=(
            "FastAPI + LangGraph + SSE + PostgreSQL + Qdrant 鐨勭綉椤电 AI 鍔╂墜"
        ),
        lifespan=lifespan,
    )
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.public_message}},
        )

    @app.get("/api/health", tags=["system"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/settings/prompt", tags=["settings"])
    async def get_system_prompt(request: Request) -> JSONResponse:
        store: PromptStore = request.app.state.container.prompt_store
        personality = await store.get_personality()
        return JSONResponse({
            "prompt": personality,
            "default": DEFAULT_PERSONALITY_PROMPT,
            "is_custom": store.is_custom,
            "length": len(personality),
        })

    @app.put("/api/settings/prompt", tags=["settings"])
    async def update_system_prompt(payload: PromptUpdate, request: Request) -> JSONResponse:
        store: PromptStore = request.app.state.container.prompt_store
        try:
            new_prompt = await store.set_personality(payload.prompt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({
            "prompt": new_prompt,
            "is_custom": True,
            "length": len(new_prompt),
        })

    @app.delete("/api/settings/prompt", tags=["settings"])
    async def reset_system_prompt(request: Request) -> JSONResponse:
        store: PromptStore = request.app.state.container.prompt_store
        personality = await store.reset_personality()
        return JSONResponse({
            "prompt": personality,
            "is_custom": False,
            "length": len(personality),
        })

    @app.get(
        "/api/settings/branding",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def get_branding(request: Request) -> BrandingOut:
        client_id = _client_id(request)
        branding = await request.app.state.container.branding_store.get(client_id)
        return _branding_out(branding, client_id)

    @app.put(
        "/api/settings/branding",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def update_branding(
        payload: BrandingUpdate,
        request: Request,
    ) -> BrandingOut:
        client_id = _client_id(request)
        try:
            branding = await request.app.state.container.branding_store.set_name(
                client_id,
                payload.name
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _branding_out(branding, client_id)

    @app.put(
        "/api/settings/branding/background",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def update_branding_background(
        payload: BackgroundUpdate,
        request: Request,
    ) -> BrandingOut:
        client_id = _client_id(request)
        try:
            branding = await request.app.state.container.branding_store.set_background(
                client_id,
                payload.enabled,
                payload.background_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _branding_out(branding, client_id)

    @app.post(
        "/api/settings/branding/logo",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def upload_branding_logo(
        request: Request,
        file: UploadFile = File(...),
    ) -> BrandingOut:
        state = request.app.state.container
        client_id = _client_id(request)
        maximum = 5 * 1024 * 1024
        content = await file.read(maximum + 1)
        await file.close()
        if len(content) > maximum:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "logo_too_large",
                    "message": "助手图片不能超过 5 MB",
                },
            )
        try:
            stored = await state.media_storage.save_image(content)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_logo",
                    "message": str(exc),
                },
            ) from exc
        try:
            branding = await state.branding_store.set_logo(
                client_id,
                stored.filename,
            )
        except Exception:
            await state.media_storage.delete("branding", stored.filename)
            raise
        return _branding_out(branding, client_id)

    @app.get("/api/settings/branding/logo", tags=["settings"])
    async def get_branding_logo(request: Request) -> FileResponse:
        client_id = _client_id(request, allow_query=True)
        branding = await request.app.state.container.branding_store.get(client_id)
        if not branding.logo_file:
            raise HTTPException(status_code=404, detail="助手图片不存在")
        target = request.app.state.container.media_storage.media_path(
            "branding",
            branding.logo_file,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="助手图片不存在")
        return FileResponse(
            target,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.delete(
        "/api/settings/branding/logo",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def reset_branding_logo(request: Request) -> BrandingOut:
        state = request.app.state.container
        client_id = _client_id(request)
        previous = await state.branding_store.get(client_id)
        branding = await state.branding_store.clear_logo(client_id)
        await state.media_storage.delete("branding", previous.logo_file)
        return _branding_out(branding, client_id)

    @app.post(
        "/api/settings/branding/background/video",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def upload_background_video(
        request: Request,
        file: UploadFile = File(...),
    ) -> BrandingOut:
        state = request.app.state.container
        client_id = _client_id(request)
        try:
            stored = await state.media_storage.save_video(file)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_video", "message": str(exc)},
            ) from exc
        try:
            branding = await state.branding_store.set_video(
                client_id,
                stored.filename,
            )
        except Exception:
            await state.media_storage.delete("backgrounds", stored.filename)
            raise
        return _branding_out(branding, client_id)

    @app.get("/api/settings/branding/background/video", tags=["settings"])
    async def get_background_video(request: Request) -> FileResponse:
        client_id = _client_id(request, allow_query=True)
        branding = await request.app.state.container.branding_store.get(client_id)
        if not branding.background_video_file:
            raise HTTPException(status_code=404, detail="背景视频不存在")
        target = request.app.state.container.media_storage.media_path(
            "backgrounds",
            branding.background_video_file,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="背景视频不存在")
        suffix = target.suffix.lower()
        media_type = "video/webm" if suffix == ".webm" else "video/mp4"
        return FileResponse(
            target,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "Accept-Ranges": "bytes",
            },
        )

    @app.get(
        "/api/settings/branding/media",
        response_model=list[BrandingMediaOut],
        tags=["settings"],
    )
    async def list_branding_media(request: Request) -> list[BrandingMediaOut]:
        state = request.app.state.container
        client_id = _client_id(request)
        branding = await state.branding_store.get(client_id)
        media: list[BrandingMediaOut] = []
        for kind, directory in (
            ("image", "branding"),
            ("video", "backgrounds"),
        ):
            for item in state.media_storage.list_media(directory):
                filename = str(item["filename"])
                usage: list[str] = []
                if kind == "image":
                    if filename == branding.logo_file:
                        usage.append("logo")
                    if (
                        filename == branding.logo_file
                        and branding.background_type == "image"
                    ):
                        usage.append("background")
                elif filename == branding.background_video_file:
                    usage.append("background")
                media.append(
                    BrandingMediaOut(
                        kind=kind,  # type: ignore[arg-type]
                        filename=filename,
                        url=(
                            f"/api/settings/branding/media/{kind}/{filename}"
                            f"?{urlencode({'client_id': client_id})}"
                        ),
                        size_bytes=int(item["size_bytes"]),
                        created_at=item["created_at"],  # type: ignore[arg-type]
                        usage=usage,
                    )
                )
        media.sort(key=lambda item: item.created_at, reverse=True)
        return media

    @app.post(
        "/api/settings/branding/media/use",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def use_branding_media(
        payload: BrandingMediaUse,
        request: Request,
    ) -> BrandingOut:
        state = request.app.state.container
        client_id = _client_id(request)
        directory = "branding" if payload.kind == "image" else "backgrounds"
        if state.media_storage.media_path(directory, payload.filename) is None:
            raise HTTPException(status_code=404, detail="媒体文件不存在")
        if payload.kind == "video":
            branding = await state.branding_store.set_video(
                client_id,
                payload.filename,
            )
        else:
            branding = await state.branding_store.set_logo(
                client_id,
                payload.filename,
            )
            if payload.target == "background":
                branding = await state.branding_store.set_background(
                    client_id,
                    True,
                    "image",
                )
        return _branding_out(branding, client_id)

    @app.get("/api/settings/branding/media/{kind}/{filename}", tags=["settings"])
    async def get_branding_media(
        kind: str,
        filename: str,
        request: Request,
    ) -> FileResponse:
        _client_id(request, allow_query=True)
        directory = "branding" if kind == "image" else "backgrounds"
        target = request.app.state.container.media_storage.media_path(
            directory,
            filename,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="媒体文件不存在")
        extension = target.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".mp4": "video/mp4",
        }.get(extension, "application/octet-stream")
        return FileResponse(
            target,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "Accept-Ranges": "bytes",
            },
        )

    @app.delete(
        "/api/settings/branding/media/{kind}/{filename}",
        response_model=BrandingOut,
        tags=["settings"],
    )
    async def delete_branding_media(
        kind: str,
        filename: str,
        request: Request,
    ) -> BrandingOut:
        state = request.app.state.container
        client_id = _client_id(request)
        directory = "branding" if kind == "image" else "backgrounds"
        if state.media_storage.media_path(directory, filename) is None:
            raise HTTPException(status_code=404, detail="媒体文件不存在")
        branding = await state.branding_store.get(client_id)
        if kind == "image" and filename == branding.logo_file:
            branding = await state.branding_store.clear_logo(client_id)
        elif kind == "video" and filename == branding.background_video_file:
            branding = await state.branding_store.clear_video(client_id)
        await state.media_storage.delete(directory, filename)
        return _branding_out(branding, client_id)

    @app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        state = request.app.state.container
        client_id, tenant_id, user_id = _identity(request)
        start = await state.chat_repository.begin_chat(
            payload.message,
            payload.session_id,
            client_id,
            tenant_id,
            user_id,
        )

        async def ignore_event(_: dict[str, Any]) -> None:
            return None

        result = await state.agent.run(
            question=payload.message,
            history=start.history,
            emit=ignore_event,
            client_id=client_id,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=str(start.session_id),
            enable_web_search=payload.enable_web_search,
            show_sources=payload.show_sources,
        )
        answer = result.get("answer", "")
        if not answer:
            from app.errors import ModelUpstreamError

            raise ModelUpstreamError("模型返回了空响应，请稍后重试")
        try:
            await state.chat_repository.finish_chat(start.session_id, answer)
        except Exception:
            logger.exception(
                "Failed to persist assistant message session_id=%s",
                start.session_id,
            )
        _schedule_memory(
            state,
            session_id=start.session_id,
            client_id=client_id,
            tenant_id=tenant_id,
            user_id=user_id,
            user_message=payload.message,
            assistant_answer=answer,
        )
        return ChatResponse(
            answer=answer,
            session_id=start.session_id,
            memories_used=len(result.get("memories", [])),
        )

    @app.post("/api/chat/stream", tags=["chat"])
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        state = request.app.state.container
        client_id, tenant_id, user_id = _identity(request)
        logger.info(
            "Chat request session_id=%s enable_web_search=%s",
            payload.session_id,
            payload.enable_web_search,
        )
        start = await state.chat_repository.begin_chat(
            payload.message,
            payload.session_id,
            client_id,
            tenant_id,
            user_id,
        )

        async def event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            answer = ""
            sources: list[dict[str, Any]] = []
            memories: list[dict[str, Any]] = []
            recalled: list[dict[str, Any]] = []
            assistant_message_id: UUID | None = None

            async def emit(event: dict[str, Any]) -> None:
                await queue.put(event)

            task = asyncio.create_task(
                state.agent.run(
                    question=payload.message,
                    history=start.history,
                    emit=emit,
                    client_id=client_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=str(start.session_id),
                    enable_web_search=payload.enable_web_search,
                    show_sources=payload.show_sources,
                )
            )
            yield _sse(
                {
                    "type": "start",
                    "session_id": str(start.session_id),
                }
            )
            try:
                while True:
                    if task.done() and queue.empty():
                        break
                    queue_task = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait(
                        {queue_task, task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if queue_task in done:
                        event = queue_task.result()
                        yield _sse(event)
                    else:
                        queue_task.cancel()
                        await asyncio.gather(queue_task, return_exceptions=True)

                result = await task
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                memories = result.get("memories", [])
                recalled = result.get("recalled_messages", [])
                if not answer:
                    from app.errors import ModelUpstreamError

                    raise ModelUpstreamError("模型返回了空响应，请稍后重试")
                try:
                    assistant_message_id = await state.chat_repository.finish_chat(
                        start.session_id, answer
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist streamed assistant message session_id=%s",
                        start.session_id,
                    )
                _schedule_memory(
                    state,
                    session_id=start.session_id,
                    client_id=client_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_message=payload.message,
                    assistant_answer=answer,
                )
                yield _sse(
                    {
                        "type": "done",
                        "status": "ok",
                        "session_id": str(start.session_id),
                        "assistant_message_id": (
                            str(assistant_message_id)
                            if assistant_message_id is not None
                            else None
                        ),
                        "sources": sources,
                        "memories": memories,
                        "recalled_conversations": recalled,
                        "memory_write_queued": True,
                    }
                )
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                logger.info(
                    "SSE client disconnected session_id=%s", start.session_id
                )
                raise
            except AppError as exc:
                logger.warning(
                    "SSE application error code=%s session_id=%s",
                    exc.code,
                    start.session_id,
                )
                yield _sse(
                    {
                        "type": "error",
                        "code": exc.code,
                        "message": exc.public_message,
                    }
                )
                yield _sse(
                    {
                        "type": "done",
                        "status": "error",
                        "session_id": str(start.session_id),
                        "sources": sources,
                        "memories": memories,
                        "recalled_conversations": recalled,
                    }
                )
            except Exception:
                logger.exception(
                    "Unexpected SSE failure session_id=%s", start.session_id
                )
                yield _sse(
                    {
                        "type": "error",
                        "code": "internal_error",
                        "message": "服务处理消息时发生异常，请查看后端日志",
                    }
                )
                yield _sse(
                    {
                        "type": "done",
                        "status": "error",
                        "session_id": str(start.session_id),
                        "sources": sources,
                        "memories": memories,
                        "recalled_conversations": recalled,
                    }
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get(
        "/api/sessions",
        response_model=list[SessionSummary],
        tags=["sessions"],
    )
    async def list_sessions(request: Request) -> list[SessionSummary]:
        client_id, tenant_id, user_id = _identity(request)
        records = await request.app.state.container.chat_repository.list_sessions(
            client_id,
            tenant_id,
            user_id,
        )
        return [SessionSummary(**record) for record in records]

    @app.get(
        "/api/sessions/{session_id}",
        response_model=SessionDetail,
        tags=["sessions"],
    )
    async def get_session(session_id: UUID, request: Request) -> SessionDetail:
        client_id, tenant_id, user_id = _identity(request)
        record = await request.app.state.container.chat_repository.get_session(
            session_id,
            client_id,
            tenant_id,
            user_id,
        )
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": "会话不存在"},
            )
        return SessionDetail(**record)

    @app.delete("/api/sessions/{session_id}", tags=["sessions"])
    async def delete_session(session_id: UUID, request: Request) -> dict[str, bool]:
        client_id, tenant_id, user_id = _identity(request)
        deleted = (
            await request.app.state.container.chat_repository.delete_session(
                session_id,
                client_id,
                tenant_id,
                user_id,
            )
        )
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": "会话不存在"},
            )
        return {"deleted": True}

    @app.get(
        "/api/memories",
        response_model=list[MemoryFactOut],
        tags=["memory"],
    )
    async def list_memories(request: Request) -> list[MemoryFactOut]:
        client_id, tenant_id, user_id = _identity(request)
        facts = await request.app.state.container.memory_repository.list_active(
            client_id,
            tenant_id,
            user_id,
        )
        return [MemoryFactOut(**fact.__dict__) for fact in facts]

    @app.delete("/api/memories/{fact_id}", tags=["memory"])
    async def delete_memory(fact_id: UUID, request: Request) -> dict[str, bool]:
        _client, tenant_id, user_id = _identity(request)
        deleted = await request.app.state.container.long_term_memory.delete(
            tenant_id,
            user_id,
            fact_id,
        )
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={"code": "memory_not_found", "message": "记忆不存在"},
            )
        return {"deleted": True}

    @app.post(
        "/api/documents/upload",
        response_model=DocumentUploadResponse,
        tags=["knowledge"],
    )
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
    ) -> DocumentUploadResponse:
        state = request.app.state.container
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()
        if extension not in {".md", ".markdown", ".txt", ".pdf"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "unsupported_file",
                    "message": "仅支持 Markdown、TXT 和 PDF 文档",
                },
            )
        maximum = state.settings.max_upload_mb * 1024 * 1024
        content = await file.read(maximum + 1)
        await file.close()
        if len(content) > maximum:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "file_too_large",
                    "message": f"文件不能超过 {state.settings.max_upload_mb} MB",
                },
            )
        try:
            document, chunk_count = await state.documents.upload(
                filename,
                file.content_type or "application/octet-stream",
                content,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_document", "message": str(exc)},
            ) from exc
        return DocumentUploadResponse(
            document=_document_out(document),
            indexed_chunks=chunk_count,
        )

    @app.get(
        "/api/documents",
        response_model=list[DocumentOut],
        tags=["knowledge"],
    )
    async def list_documents(request: Request) -> list[DocumentOut]:
        records = (
            await request.app.state.container.document_repository.list()
        )
        return [_document_out(document) for document in records]

    @app.get(
        "/api/rag/search",
        response_model=SearchResponse,
        tags=["knowledge"],
    )
    async def rag_search(q: str, request: Request) -> SearchResponse:
        query = q.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail={"code": "empty_query", "message": "妫€绱㈣瘝涓嶈兘涓虹┖"},
            )
        results = await request.app.state.container.rag.search(query)
        return SearchResponse(query=query, results=results)

    @app.delete("/api/documents/{document_id}", tags=["knowledge"])
    async def delete_document(
        document_id: UUID, request: Request
    ) -> dict[str, bool]:
        deleted = await request.app.state.container.documents.delete(document_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={"code": "document_not_found", "message": "文档不存在"},
            )
        return {"deleted": True}

    @app.get("/api/audio/engines", tags=["audio"])
    async def list_engines(request: Request) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        return JSONResponse({
            "configured": tts.configured,
            "active_engine": tts.active_engine,
            "engines": tts.available_engines,
        })

    @app.post("/api/audio/engines/cosyvoice/download", tags=["audio"])
    async def download_cosyvoice(request: Request) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        importable = tts._check_cosyvoice_importable()
        if not importable:
            raise HTTPException(
                status_code=400,
                detail={"code": "cosyvoice_not_installed", "message": "CosyVoice2 依赖未安装"},
            )
        try:
            await tts.download_cosyvoice()
            return JSONResponse({"downloaded": True})
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "download_failed", "message": str(exc)},
            )

    @app.get("/api/audio/voices", tags=["audio"])
    async def list_voices(request: Request) -> JSONResponse:
        from app.services.audio import EDGE_CN_VOICES

        tts: TTSService = request.app.state.container.tts
        voices = []
        for k, v in tts.available_voices.items():
            engine = "edge_tts" if k in EDGE_CN_VOICES else "rvc"
            voices.append({"id": k, "name": v, "engine": engine})
        return JSONResponse({
            "configured": tts.configured,
            "default_voice": tts.default_voice if tts.configured else "",
            "voices": voices,
            "engine": tts.active_engine,
            "rvc_enabled": tts.rvc_enabled,
        })

    @app.get("/api/audio/prompts", tags=["audio"])
    async def list_voice_prompts(request: Request) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        return JSONResponse({"prompts": tts.prompts.list_prompts()})

    @app.post("/api/audio/prompts", tags=["audio"])
    async def upload_voice_prompt(
        request: Request,
        file: UploadFile = File(...),
        name: str = Form(""),
        text: str = Form(""),
        duration_limit: float = Form(15.0),
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        content = await file.read()
        try:
            prompt, warning = await tts.prompts.add(
                filename=file.filename or "prompt.wav",
                content=content,
                name=name,
                text=text,
                duration_limit=duration_limit,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_prompt", "message": str(exc)},
            ) from exc
        warnings = [warning] if warning else []
        try:
            prepared = await tts.prepare_voice_prompt(prompt["id"])
            if not prepared:
                warnings.append("CosyVoice2 尚未就绪，首次合成时会自动缓存音色")
        except Exception as exc:
            logger.warning("Voice prompt pre-cache failed", exc_info=True)
            warnings.append(f"音色缓存将在首次合成时重试：{exc}")
        return JSONResponse(
            {
                "prompt": prompt,
                "recognition_warning": "；".join(warnings) if warnings else None,
            }
        )

    @app.get("/api/audio/prompts/{prompt_id}/audio", tags=["audio"])
    async def get_voice_prompt_audio(
        request: Request,
        prompt_id: str,
    ) -> FileResponse:
        tts: TTSService = request.app.state.container.tts
        try:
            path = tts.prompts.audio_path(prompt_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    @app.put("/api/audio/prompts/{prompt_id}", tags=["audio"])
    async def update_voice_prompt(
        request: Request,
        prompt_id: str,
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="请求体必须是 JSON")
        try:
            prompt = tts.prompts.update(
                prompt_id,
                name=payload.get("name"),
                text=payload.get("text"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        tts.invalidate_voice_prompt(prompt_id)
        return JSONResponse({"prompt": prompt})

    @app.post("/api/audio/prompts/{prompt_id}/transcribe", tags=["audio"])
    async def transcribe_voice_prompt(
        request: Request,
        prompt_id: str,
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        try:
            prompt = await tts.prompts.transcribe(prompt_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "recognition_failed", "message": str(exc)},
            ) from exc
        tts.invalidate_voice_prompt(prompt_id)
        try:
            await tts.prepare_voice_prompt(prompt_id)
        except Exception:
            logger.warning("Voice prompt re-cache failed", exc_info=True)
        return JSONResponse({"prompt": prompt})

    @app.delete("/api/audio/prompts/{prompt_id}", tags=["audio"])
    async def delete_voice_prompt(
        request: Request,
        prompt_id: str,
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        try:
            tts.prompts.delete(prompt_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        tts.invalidate_voice_prompt(prompt_id)
        return JSONResponse({"deleted": True})

    @app.post("/api/audio/tts/stream", tags=["audio"])
    async def text_to_speech_stream(request: Request) -> StreamingResponse:
        state = request.app.state.container
        tts: TTSService = state.tts
        if not tts.configured:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "tts_not_configured",
                    "message": "TTS 服务未配置",
                },
            )
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_json", "message": "请求体必须是 JSON"},
            )
        raw_message_id = str(payload.get("message_id", "")).strip()
        if not raw_message_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "message_id_required",
                    "message": "缺少 message_id",
                },
            )
        try:
            message_id = UUID(raw_message_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_message_id",
                    "message": "message_id 必须是 UUID",
                },
            ) from exc
        client_id, tenant_id, user_id = _identity(request)
        text = await state.chat_repository.get_message_content(
            message_id,
            client_id=client_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if text is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "message_not_found",
                    "message": "助手消息不存在或不属于当前用户",
                },
            )
        text = text.strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail={"code": "empty_message", "message": "助手消息内容为空"},
            )

        engine = str(payload.get("engine", "")).strip() or None
        edge_voice = str(payload.get("edge_voice", "")).strip() or None
        requested_rvc = payload.get("rvc_enabled")
        rvc_enabled = requested_rvc if isinstance(requested_rvc, bool) else None
        rvc_model = str(payload.get("rvc_model", "")).strip() or None
        rvc_index = str(payload.get("rvc_index", "")).strip() or None
        prompt_id = str(payload.get("prompt_id", "")).strip() or None
        try:
            rvc_index_rate = float(payload.get("rvc_index_rate", 0.6))
            rvc_protect = float(payload.get("rvc_protect", 0.4))
            rvc_rms_mix_rate = float(payload.get("rvc_rms_mix_rate", 0.25))
            rvc_f0_up_key = int(payload.get("rvc_f0_up_key", 0))
            tts_speed = float(payload.get("tts_speed", 1.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_rvc_parameters",
                    "message": "RVC 参数必须是数字",
                },
            ) from exc
        if not (
            -12 <= rvc_f0_up_key <= 12
            and 0 <= rvc_index_rate <= 1
            and 0 <= rvc_protect <= 0.5
            and 0 <= rvc_rms_mix_rate <= 1
            and 0.7 <= tts_speed <= 1.3
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_rvc_parameters",
                    "message": "RVC 参数超出允许范围",
                },
            )

        async def audio_events() -> AsyncIterator[str]:
            try:
                segment_index = 0
                async for segment, sample_rate, audio, used_engine in tts.synthesize_pcm_stream(
                    text,
                    voice=edge_voice,
                    engine=engine,
                    rvc_enabled=rvc_enabled,
                    rvc_model=rvc_model,
                    rvc_index=rvc_index,
                    prompt_id=prompt_id,
                    rvc_index_rate=rvc_index_rate,
                    rvc_protect=rvc_protect,
                    rvc_rms_mix_rate=rvc_rms_mix_rate,
                    rvc_f0_up_key=rvc_f0_up_key,
                    tts_speed=tts_speed,
                ):
                    yield json.dumps(
                        {
                            "type": "audio",
                            "index": segment_index,
                            "text": segment,
                            "engine": used_engine,
                            "format": "pcm_s16le",
                            "sample_rate": sample_rate,
                            "channels": 1,
                            "data": base64.b64encode(audio).decode("ascii"),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n"
                    segment_index += 1
                yield json.dumps(
                    {
                        "type": "done",
                        "engine": tts.active_engine,
                    },
                    separators=(",", ":"),
                ) + "\n"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Streaming TTS failed")
                yield json.dumps(
                    {
                        "type": "error",
                        "code": "tts_stream_failed",
                        "message": str(exc) or "语音生成失败",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n"

        return StreamingResponse(
            audio_events(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/audio/rvc/runtime", tags=["audio"])
    async def set_rvc_runtime(request: Request) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_json", "message": "请求体必须是 JSON"},
            )
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_enabled", "message": "enabled 必须是布尔值"},
            )
        active = await asyncio.to_thread(tts.set_rvc_enabled, enabled)
        if active and tts._rvc_engine is None:
            await asyncio.to_thread(tts._init_rvc)
        return JSONResponse({"enabled": active})

    @app.get("/api/audio/rvc/models", tags=["audio"])
    async def list_rvc_models(request: Request) -> JSONResponse:
        from app.services.audio import _scan_rvc_models

        tts: TTSService = request.app.state.container.tts
        model_dir = tts.rvc_model_dir
        index_dir = tts.rvc_index_dir

        models: list[dict] = []
        for pth in _scan_rvc_models(model_dir):
            stem = pth.stem
            indexes: list[dict] = []
            selected_index: str | None = None
            if index_dir.exists():
                all_index_files = sorted(
                    index_dir.glob("*.index"),
                    key=lambda path: path.name.lower(),
                )
                index_files = sorted(
                    index_dir.glob(f"{stem}*.index"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if index_files:
                    selected_index = index_files[0].name
                for idx_file in all_index_files:
                    indexes.append({
                        "filename": idx_file.name,
                        "size_bytes": idx_file.stat().st_size,
                    })
            models.append({
                "name": stem,
                "pth_filename": pth.name,
                "pth_size_bytes": pth.stat().st_size,
                "indexes": indexes,
                "selected_index": selected_index,
            })
        model_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        return JSONResponse({"models": models})

    @app.post("/api/audio/rvc/preload", tags=["audio"])
    async def preload_rvc_model(
        request: Request,
        model_name: str = "",
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        safe_name = Path(model_name).name if model_name else ""
        if model_name and (not safe_name or safe_name != model_name):
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_model_name", "message": "模型名称无效"},
            )
        try:
            loaded = await tts.preload_rvc_model(safe_name or None)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "rvc_preload_failed", "message": str(exc)},
            ) from exc
        return JSONResponse({"preloaded": True, "model_name": loaded})

    @app.post("/api/audio/rvc/models", tags=["audio"])
    async def upload_rvc_model(
        request: Request,
        model: UploadFile = File(...),
        index: UploadFile | None = File(None),
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        if not tts.configured:
            raise HTTPException(
                status_code=503,
                detail={"code": "tts_not_configured", "message": "TTS 服务未配置，无法上传模型"},
            )

        _validate_rvc_extension(model.filename, ".pth")
        contents = await model.read()
        if not contents:
            raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "模型文件不能为空"})

        model_dir = tts.rvc_model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        pth_path = model_dir / Path(model.filename).name
        pth_path.write_bytes(contents)

        index_saved: str | None = None
        if index is not None and index.filename:
            _validate_rvc_extension(index.filename, ".index")
            index_contents = await index.read()
            if index_contents:
                index_dir = tts.rvc_index_dir
                index_dir.mkdir(parents=True, exist_ok=True)
                idx_path = index_dir / Path(index.filename).name
                idx_path.write_bytes(index_contents)
                index_saved = idx_path.name

        logger.info("RVC model uploaded: %s (index=%s)", pth_path.name, index_saved or "none")
        return JSONResponse({
            "model_name": pth_path.stem,
            "index_saved": index_saved,
        })

    @app.post("/api/audio/rvc/models/{model_name}/index", tags=["audio"])
    async def upload_rvc_index(
        request: Request,
        model_name: str,
        index: UploadFile = File(...),
    ) -> JSONResponse:
        tts: TTSService = request.app.state.container.tts
        safe_name = Path(model_name).name
        if not safe_name or safe_name != model_name:
            raise HTTPException(status_code=400, detail={"code": "invalid_name", "message": "无效的模型名称"})

        model_dir = tts.rvc_model_dir
        pth_path = model_dir / f"{safe_name}.pth"
        if not pth_path.exists():
            raise HTTPException(
                status_code=404,
                detail={"code": "model_not_found", "message": f"模型 {safe_name} 不存在，请先上传对应的 .pth 文件"},
            )

        _validate_rvc_extension(index.filename, ".index")
        contents = await index.read()
        if not contents:
            raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "索引文件不能为空"})

        index_dir = tts.rvc_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        idx_path = index_dir / Path(index.filename).name
        idx_path.write_bytes(contents)

        logger.info("RVC index uploaded for model %s: %s", safe_name, idx_path.name)
        return JSONResponse({
            "model_name": safe_name,
            "index_saved": idx_path.name,
        })

    @app.delete("/api/audio/rvc/models/{model_name}", tags=["audio"])
    async def delete_rvc_model(
        request: Request,
        model_name: str,
    ) -> JSONResponse:
        from app.services.audio import _scan_rvc_models

        tts: TTSService = request.app.state.container.tts
        safe_name = Path(model_name).name
        if not safe_name or safe_name != model_name:
            raise HTTPException(status_code=400, detail={"code": "invalid_name", "message": "无效的模型名称"})

        model_dir = tts.rvc_model_dir
        pth_path = model_dir / f"{safe_name}.pth"
        if not pth_path.exists():
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "模型文件不存在"})

        pth_path.unlink()
        deleted_indexes: list[str] = []
        index_dir = tts.rvc_index_dir
        if index_dir.exists():
            for idx_file in sorted(index_dir.glob(f"{safe_name}*.index")):
                idx_file.unlink()
                deleted_indexes.append(idx_file.name)

        logger.info("RVC model deleted: %s (indexes=%s)", safe_name, deleted_indexes or "none")
        return JSONResponse({
            "deleted": True,
            "model_name": safe_name,
            "deleted_indexes": deleted_indexes,
        })

    @app.post("/api/audio/rvc/refresh", tags=["audio"])
    async def refresh_rvc_voices(request: Request) -> JSONResponse:
        from app.services.audio import _scan_rvc_models

        tts: TTSService = request.app.state.container.tts
        model_dir = tts.rvc_model_dir
        models = _scan_rvc_models(model_dir)

        if not tts.configured:
            return JSONResponse({
                "configured": False,
                "voices": [],
                "rvc_models_found": len(models),
            })

        voices = [
            {"id": pth.stem, "name": pth.stem}
            for pth in models
        ]
        if not voices:
            voices = [{"id": "cosyvoice", "name": "CosyVoice 合成音色"}]

        tts._warmed_up = False
        logger.info("RVC models refreshed, found %d model(s)", len(models))
        return JSONResponse({
            "configured": True,
            "default_voice": tts.default_voice if voices else "",
            "voices": voices,
        })

    return app


def _validate_rvc_extension(filename: str, expected: str) -> None:
    if not filename:
        raise HTTPException(status_code=400, detail={"code": "missing_file", "message": "缺少文件"})
    suf = Path(filename).suffix.lower()
    if suf != expected:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_extension", "message": f"只接受 {expected} 文件，得到了 {suf}"},
        )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime_settings = get_settings()
    uvicorn.run(
        app,
        host=runtime_settings.api_host,
        port=runtime_settings.api_port,
    )
