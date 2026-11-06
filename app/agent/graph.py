from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.errors import ModelUpstreamError
from app.services.conversation_memory import ConversationRecall
from app.services.llm import LLMClient
from app.services.memory import LongTermMemory, format_memory_context
from app.services.crawler import WebCrawlerService
from app.services.rag import RagService
from app.services.web_search import WebSearchService

logger = logging.getLogger(__name__)
Emit = Callable[[dict[str, Any]], Awaitable[None]]


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    client_id: str
    tenant_id: str
    user_id: str
    session_id: str | None
    enable_web_search: bool
    show_sources: bool
    route: str
    tool_result: str
    tool_error: str
    sources: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    memories: list[dict[str, Any]]
    deleted_memories: list[dict[str, Any]]
    recalled_messages: list[dict[str, Any]]
    rag_error: str
    answer: str
    error: str


def _emit_from_config(
    config: RunnableConfig,
) -> Emit:
    configurable = config.get("configurable") or {}
    emit = configurable.get("emit")
    if emit is None:
        async def noop(_: dict[str, Any]) -> None:
            return None

        return noop
    return emit


class AssistantGraph:
    def __init__(
        self,
        llm: LLMClient,
        rag: RagService,
        long_term_memory: LongTermMemory | None = None,
        web_search: WebSearchService | None = None,
        conversation_recall: ConversationRecall | None = None,
        web_crawler: WebCrawlerService | None = None,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.long_term_memory = long_term_memory
        self.web_search = web_search
        self.conversation_recall = conversation_recall
        self.web_crawler = web_crawler
        graph = StateGraph(AgentState)
        graph.add_node("classify", self._classify)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("time_tool", self._time_tool)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("web_search", self._web_search_node)
        graph.add_node("crawl", self._crawl)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "load_memory")
        graph.add_conditional_edges(
            "load_memory",
            self._route,
            {
                "time": "time_tool",
                "rag": "retrieve",
                "chat": "answer",
            },
        )
        graph.add_edge("time_tool", "web_search")
        graph.add_edge("retrieve", "web_search")
        graph.add_edge("web_search", "crawl")
        graph.add_edge("crawl", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    @staticmethod
    def _route(state: AgentState) -> str:
        return state.get("route", "chat")

    async def _classify(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        question = state["question"]
        time_pattern = re.compile(
            r"(当前时间|现在几点|今天几号|今天日期|星期几|"
            r"\bwhat time\b|\bcurrent time\b|\btoday'?s date\b)",
            re.IGNORECASE,
        )
        route = "time" if time_pattern.search(question) else "rag"
        logger.info("Agent entered node=classify route=%s", route)
        await emit({"type": "status", "stage": "classify", "route": route})
        return {"route": route}

    async def _load_memory(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        active: list[dict[str, Any]] = []
        deleted: list[dict[str, Any]] = []
        if self.long_term_memory is not None:
            active, deleted = await self.long_term_memory.retrieve_safe(
                state.get("tenant_id", "default"),
                state.get("user_id", state.get("client_id", "default")),
                state["question"],
            )
        recalled: list[dict[str, Any]] = []
        wants_history = bool(
            re.search(
                r"(之前|上次|曾经|继续|记得|往期|先前|"
                r"earlier|previous|last time|remember)",
                state["question"],
                re.IGNORECASE,
            )
        )
        if (
            self.conversation_recall is not None
            and state.get("session_id")
            and (not active or wants_history)
        ):
            recalled = await self.conversation_recall.recall_safe(
                client_id=state.get("client_id", "default"),
                tenant_id=state.get("tenant_id", "default-tenant"),
                user_id=state.get("user_id", state.get("client_id", "default")),
                current_session_id=UUID(state["session_id"]),
                query=state["question"],
            )
        logger.info(
            "Agent entered node=load_memory active=%s deleted=%s recalled=%s",
            len(active),
            len(deleted),
            len(recalled),
        )
        await emit(
            {
                "type": "status",
                "stage": "load_memory",
                "memory_count": len(active),
                "deleted_memory_count": len(deleted),
                "recalled_conversation_count": len(recalled),
                "message": (
                    f"长期记忆为空，已检索 {len(recalled)} 条往期对话"
                    if recalled and not active
                    else (
                        f"已参考 {len(recalled)} 条相关往期对话"
                        if recalled
                        else "未检索到相关长期记忆"
                    )
                ),
            }
        )
        return {
            "memories": active,
            "deleted_memories": deleted,
            "recalled_messages": recalled,
        }

    async def _time_tool(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        try:
            now = datetime.now(timezone(timedelta(hours=8)))
            result = now.strftime("%Y年%m月%d日 %H:%M:%S，星期")
            weekday = "一二三四五六日"[now.weekday()]
            result += weekday + "（Asia/Shanghai）"
            logger.info("Agent tool=get_current_time result=%s", result)
            await emit(
                {
                    "type": "status",
                    "stage": "time_tool",
                    "message": "已执行当前时间工具",
                }
            )
            return {"tool_result": result, "tool_error": ""}
        except Exception as exc:
            logger.exception("Agent tool get_current_time failed")
            await emit(
                {
                    "type": "status",
                    "stage": "time_tool",
                    "message": "时间工具执行失败，将继续普通回答",
                }
            )
            return {
                "tool_result": "",
                "tool_error": f"当前时间工具执行失败：{type(exc).__name__}",
            }

    async def _retrieve(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        logger.info("Agent entered node=retrieve")
        sources = await self.rag.search_safe(state["question"])
        logger.info("Agent RAG retrieval source_count=%s", len(sources))
        await emit(
            {
                "type": "status",
                "stage": "retrieve",
                "source_count": len(sources),
                "message": (
                    f"已检索到 {len(sources)} 条知识库片段"
                    if sources
                    else "未启用或未检索到知识库片段，继续普通聊天"
                ),
            }
        )
        return {"sources": sources, "rag_error": ""}

    async def _web_search_node(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        if not state.get("enable_web_search") or self.web_search is None:
            logger.info("Agent web_search skipped (disabled or no service)")
            await emit(
                {
                    "type": "status",
                    "stage": "web_search",
                    "message": "联网搜索未启用",
                }
            )
            return {"web_results": []}
        logger.info("Agent entered node=web_search")
        await emit(
            {
                "type": "status",
                "stage": "web_search",
                "message": "正在联网搜索最新信息",
            }
        )
        results = await self.web_search.search_safe(state["question"])
        logger.info("Agent web_search result_count=%s", len(results))
        await emit(
            {
                "type": "status",
                "stage": "web_search",
                "source_count": len(results),
                "message": (
                    f"联网搜索到 {len(results)} 条结果"
                    if results
                    else "联网搜索未找到相关结果"
                ),
            }
        )
        return {"web_results": results}

    async def _crawl(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        if self.web_crawler is None:
            return {"tool_result": "", "tool_error": ""}
        url = WebCrawlerService.extract_url(state.get("question", ""))
        if not url:
            logger.info("Agent crawl skipped: no URL in question")
            return {"tool_result": "", "tool_error": ""}
        logger.info("Agent entered node=crawl url=%s", url)
        await emit(
            {
                "type": "status",
                "stage": "crawl",
                "message": f"正在抓取网页 {url}",
            }
        )
        page = await self.web_crawler.fetch_safe(url)
        if page is None:
            logger.info("Agent crawl returned no content url=%s", url)
            await emit(
                {
                    "type": "status",
                    "stage": "crawl",
                    "message": "网页抓取失败或无正文内容",
                }
            )
            return {"tool_result": "", "tool_error": "网页抓取失败或无正文内容"}
        content = (
            f"[已抓取网页: {page['title'] or page['url']} | {page['url']}]\n"
            f"{page['content']}"
        )
        logger.info("Agent crawl content_chars=%s", len(page["content"]))
        await emit(
            {
                "type": "status",
                "stage": "crawl",
                "content_chars": len(page["content"]),
                "message": f"已抓取网页正文（{len(page['content'])} 字符）",
            }
        )
        return {"tool_result": content, "tool_error": ""}

    async def _answer(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        sources = state.get("sources", [])
        web_results = state.get("web_results", [])
        context_parts = []
        for source in sources:
            context_parts.append(
                f"[知识库来源: {source['filename']} | 片段 {source['chunk_index'] + 1}]\n"
                f"{source['content']}"
            )
        for item in web_results:
            context_parts.append(
                f"[联网来源: {item.get('title', '无标题')} | {item.get('url', '')}]\n"
                f"{item.get('snippet', '')}"
            )
        tool_context = state.get("tool_result", "")
        if state.get("tool_error"):
            tool_context = state["tool_error"]
        memories = state.get("memories", [])
        deleted_memories = state.get("deleted_memories", [])
        memory_context = format_memory_context(memories, deleted_memories or None)
        recalled = state.get("recalled_messages", [])
        conversation_context = "\n\n".join(
            f"[往期会话: {item.get('session_title', '未命名')} | "
            f"{item.get('role', 'user')}]\n{item.get('content', '')}"
            for item in recalled
        )

        logger.info(
            "Agent entered node=answer model=%s sources=%s web=%s "
            "memories=%s recalled=%s",
            self.llm.settings.llm_model,
            len(sources),
            len(web_results),
            len(memories),
            len(recalled),
        )
        pieces: list[str] = []
        async for delta in self.llm.stream_answer(
            message=state["question"],
            history=state.get("history", []),
            knowledge_context="\n\n".join(context_parts),
            tool_context=tool_context,
            long_term_context=memory_context,
            conversation_context=conversation_context,
        ):
            pieces.append(delta)
            await emit({"type": "delta", "content": delta})

        answer = "".join(pieces).strip()
        if not answer:
            raise ModelUpstreamError("模型返回了空响应，请稍后重试")

        if state.get("show_sources") and (sources or web_results):
            refs: list[str] = []
            if sources:
                filenames = sorted({source["filename"] for source in sources})
                missing_kb = [name for name in filenames if name not in answer]
                refs.extend(f"《{name}》" for name in missing_kb)
            if web_results:
                urls = sorted({item.get("url", "") for item in web_results if item.get("url")})
                missing_web = [u for u in urls if u and u not in answer]
                refs.extend(f"{u}" for u in missing_web)
            if refs:
                suffix = "\n\n参考来源：" + "、".join(refs[:8])
                pieces.append(suffix)
                answer += suffix
                await emit({"type": "delta", "content": suffix})

        logger.info("Agent node=answer completed answer_chars=%s", len(answer))
        return {"answer": answer}

    async def run(
        self,
        *,
        question: str,
        history: list[dict[str, str]],
        emit: Emit,
        client_id: str = "default",
        tenant_id: str = "default",
        user_id: str | None = None,
        session_id: str | None = None,
        enable_web_search: bool = False,
        show_sources: bool = False,
    ) -> AgentState:
        return await self.graph.ainvoke(
            {
                "question": question,
                "history": history,
                "client_id": client_id,
                "tenant_id": tenant_id,
                "user_id": user_id or client_id,
                "session_id": session_id,
                "enable_web_search": enable_web_search,
                "show_sources": show_sources,
            },
            config={"configurable": {"emit": emit}},
        )