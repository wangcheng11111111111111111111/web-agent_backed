from __future__ import annotations

import asyncio
import ast
import json
import logging
import math
import operator
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
from app.storage.repositories import DocumentRepository, MemoryRepository

logger = logging.getLogger(__name__)
Emit = Callable[[dict[str, Any]], Awaitable[None]]

_CALCULATOR_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_calculate(expression: str) -> float | int:
    if len(expression) > 200:
        raise ValueError("计算表达式过长")

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _CALCULATOR_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("指数过大")
            return _CALCULATOR_OPERATORS[type(node.op)](left, right)
        raise ValueError("表达式包含不支持的运算")

    result = evaluate(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("计算结果不是有限数值")
    return result


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    client_id: str
    tenant_id: str
    user_id: str
    session_id: str | None
    enable_web_search: bool
    show_sources: bool
    tool_calls: list[dict[str, Any]]
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
        document_repository: DocumentRepository | None = None,
        memory_repository: MemoryRepository | None = None,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.long_term_memory = long_term_memory
        self.web_search = web_search
        self.conversation_recall = conversation_recall
        self.web_crawler = web_crawler
        self.document_repository = document_repository
        self.memory_repository = memory_repository
        graph = StateGraph(AgentState)
        graph.add_node("select_tools", self._select_tools)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "select_tools")
        graph.add_edge("select_tools", "load_memory")
        graph.add_edge("load_memory", "execute_tools")
        graph.add_edge("execute_tools", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    async def _select_tools(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        tools = self._available_tools(state)
        available_names = [
            str(item.get("function", {}).get("name", ""))
            for item in tools
        ]
        logger.info(
            "Tool selection input enable_web_search=%s realtime=%s available=%s",
            bool(state.get("enable_web_search")),
            self._looks_realtime(state["question"]),
            available_names,
        )
        selected = await self.llm.select_tools(
            message=state["question"],
            history=state.get("history", []),
            tools=tools,
        )
        selection_error = ""
        if not selected and self._looks_realtime(state["question"]):
            if "web_search" in available_names:
                selected = [
                    {
                        "id": "realtime_guard",
                        "name": "web_search",
                        "arguments": {"query": state["question"]},
                    }
                ]
                logger.warning(
                    "LLM selected no tool for realtime question; forced web_search"
                )
            elif (
                "fetch_web_page" in available_names
                and self.web_crawler is not None
                and WebCrawlerService.extract_url(state["question"])
            ):
                selected = [
                    {
                        "id": "realtime_guard",
                        "name": "fetch_web_page",
                        "arguments": {},
                    }
                ]
            else:
                selection_error = (
                    "实时信息工具未调用：本次请求没有启用 Bing 联网搜索，"
                    "且问题中没有可抓取的 URL。请开启前端的联网开关后重试。"
                )
        names = [str(item.get("name", "")) for item in selected]
        logger.info("Agent model-selected tools=%s", names)
        await emit(
            {
                "type": "status",
                "stage": "select_tools",
                "message": (
                    f"模型选择工具：{', '.join(names)}"
                    if names
                    else "模型判断无需调用工具"
                ),
                "tool_count": len(names),
            }
        )
        return {
            "tool_calls": selected,
            "tool_error": selection_error,
        }

    @staticmethod
    def _looks_realtime(question: str) -> bool:
        return bool(
            re.search(
                r"(今天|今日|现在|最新|实时|当前|刚刚|本周|比赛|赛程|"
                r"新闻|天气|股价|汇率|比分|today|latest|current|"
                r"news|schedule|price|score)",
                question,
                re.IGNORECASE,
            )
        )

    def _available_tools(self, state: AgentState) -> list[dict[str, object]]:
        empty_parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": (
                        "获取服务端当前日期、时间和星期。仅当用户询问"
                        "当前时间、日期、星期时调用。"
                    ),
                    "parameters": empty_parameters,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "检索用户上传的本地知识库。问题可能涉及文档、"
                        "资料、项目说明或已上传内容时调用。"
                    ),
                    "parameters": empty_parameters,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": (
                        "执行确定性的基础算术，支持加减乘除、整除、取余、"
                        "幂和括号。涉及精确计算时调用。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "只包含数字和算术运算符的表达式",
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self.document_repository is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "list_knowledge_documents",
                        "description": (
                            "列出当前知识库中的文档名称、大小和切片数量。"
                            "用户询问有哪些资料或文档时需要调用。"
                        ),
                        "parameters": empty_parameters,
                    },
                }
            )
        if self.memory_repository is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "list_long_term_memories",
                        "description": (
                            "列出当前用户的长期记忆事实。用户询问"
                            "“你记得我什么”或需要核对记忆时调用。"
                        ),
                        "parameters": empty_parameters,
                    },
                }
            )
        if self.conversation_recall is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_conversation_history",
                        "description": (
                            "检索当前用户其他历史会话中的相关消息。用户"
                            "明确询问以前聊过什么或需要继续旧话题时调用。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "用于检索历史会话的关键词",
                                }
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        if self.web_crawler is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_web_page",
                        "description": (
                            "抓取用户问题中明确提供的 HTTP/HTTPS 网页并"
                            "读取正文。只有问题包含 URL 时才可调用。"
                        ),
                        "parameters": empty_parameters,
                    },
                }
            )
        if (
            self.web_search is not None
            and state.get("enable_web_search")
            and self.web_search.settings.web_search_enabled
        ):
            today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "通过 Bing 搜索最新公开信息。用户开启联网并"
                            "需要新闻、实时信息或外部资料时调用。query "
                            "应包含对象、地点、年份和日期，必要时加入英文关键词。"
                            f"当前日期是 {today}，用户说今天、昨天或明天时必须"
                            "换算为明确日期。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "优化后的 Bing 搜索词",
                                }
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return tools

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

    async def _execute_tools(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        emit = _emit_from_config(config)
        selected = state.get("tool_calls", [])[:8]
        tool_parts: list[str] = []
        tool_errors: list[str] = []
        sources: list[dict[str, Any]] = []
        web_results: list[dict[str, Any]] = []
        executed: set[tuple[str, str]] = set()

        for call in selected:
            name = str(call.get("name", "")).strip()
            arguments = call.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            execution_key = (
                name,
                json.dumps(arguments, ensure_ascii=False, sort_keys=True),
            )
            if not name or execution_key in executed:
                continue
            executed.add(execution_key)
            logger.info("Agent executing model-selected tool=%s", name)
            update: dict[str, Any]
            if name == "get_current_time":
                update = await self._time_tool(state, config)
            elif name == "search_knowledge_base":
                update = await self._retrieve(state, config)
                count = len(update.get("sources", []))
                tool_parts.append(f"知识库检索返回 {count} 条片段。")
            elif name == "web_search":
                query = (
                    str(arguments.get("query", "")).strip()
                    if isinstance(arguments, dict)
                    else ""
                )
                update = await self._web_search_node(
                    state,
                    config,
                    query=query or state["question"],
                )
                results = update.get("web_results", [])
                web_results.extend(results)
                tool_parts.append(f"Bing 联网搜索返回 {len(results)} 条结果。")
            elif name == "fetch_web_page":
                update = await self._crawl(state, config)
            elif name == "calculate":
                try:
                    expression = str(arguments.get("expression", "")).strip()
                    value = _safe_calculate(expression)
                    update = {
                        "tool_result": f"{expression} = {value:g}",
                    }
                except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
                    update = {
                        "tool_error": f"计算失败：{exc}",
                    }
            elif name == "list_knowledge_documents":
                if self.document_repository is None:
                    update = {"tool_error": "知识库文档工具不可用"}
                else:
                    try:
                        documents = await self.document_repository.list()
                        lines = [
                            (
                                f"- {doc.filename}: {doc.size_bytes} bytes, "
                                f"{doc.chunk_count} chunks"
                            )
                            for doc in documents
                        ]
                        update = {
                            "tool_result": (
                                "知识库文档：\n" + "\n".join(lines)
                                if lines
                                else "知识库中暂无文档。"
                            )
                        }
                    except Exception as exc:
                        update = {
                            "tool_error": (
                                f"知识库文档列表读取失败：{type(exc).__name__}"
                            )
                        }
            elif name == "list_long_term_memories":
                if self.memory_repository is None:
                    update = {"tool_error": "长期记忆工具不可用"}
                else:
                    try:
                        memories = await self.memory_repository.list_active(
                            state.get("client_id", "default"),
                            state.get("tenant_id", "default"),
                            state.get("user_id", state.get("client_id", "default")),
                        )
                        lines = [
                            f"- [{fact.fact_type}] {fact.content}"
                            for fact in memories
                        ]
                        update = {
                            "tool_result": (
                                "长期记忆：\n" + "\n".join(lines)
                                if lines
                                else "当前没有长期记忆。"
                            )
                        }
                    except Exception as exc:
                        update = {
                            "tool_error": (
                                f"长期记忆读取失败：{type(exc).__name__}"
                            )
                        }
            elif name == "search_conversation_history":
                if self.conversation_recall is None:
                    update = {"tool_error": "历史会话检索不可用"}
                elif not state.get("session_id"):
                    update = {"tool_error": "当前会话 ID 缺失，无法检索历史"}
                else:
                    query = str(arguments.get("query", "")).strip()
                    recalled = await self.conversation_recall.recall_safe(
                        client_id=state.get("client_id", "default"),
                        tenant_id=state.get("tenant_id", "default"),
                        user_id=state.get(
                            "user_id",
                            state.get("client_id", "default"),
                        ),
                        current_session_id=UUID(state["session_id"]),
                        query=query or state["question"],
                    )
                    lines = [
                        (
                            f"- {item.get('session_title', '未命名')} "
                            f"[{item.get('role', 'user')}]: "
                            f"{item.get('content', '')[:500]}"
                        )
                        for item in recalled
                    ]
                    update = {
                        "tool_result": (
                            "相关历史会话：\n" + "\n".join(lines)
                            if lines
                            else "未找到相关历史会话。"
                        )
                    }
            else:
                update = {
                    "tool_error": f"模型请求了不支持的工具：{name}",
                }

            if update.get("tool_result"):
                tool_parts.append(str(update["tool_result"]))
            if update.get("tool_error"):
                tool_errors.append(str(update["tool_error"]))
            sources.extend(update.get("sources", []))
            if name != "web_search":
                web_results.extend(update.get("web_results", []))

        await emit(
            {
                "type": "status",
                "stage": "execute_tools",
                "message": (
                    f"已执行 {len(executed)} 个工具"
                    if executed
                    else "无需执行工具"
                ),
                "tool_count": len(executed),
            }
        )
        return {
            "tool_result": "\n\n".join(tool_parts),
            "tool_error": "\n".join(
                item
                for item in [state.get("tool_error", ""), *tool_errors]
                if item
            ),
            "sources": sources,
            "web_results": web_results,
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
        self,
        state: AgentState,
        config: RunnableConfig,
        query: str | None = None,
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
        search_query = query or state["question"]
        results = await self.web_search.search_safe(search_query)
        enriched_count = 0
        failed_count = 0
        if results and self.web_crawler is not None:
            candidates = results[:4]
            await emit(
                {
                    "type": "status",
                    "stage": "web_search",
                    "message": f"搜索完成，正在读取 {len(candidates)} 个结果页面",
                    "source_count": len(results),
                    "query": search_query,
                }
            )
            pages = await asyncio.gather(
                *(
                    self.web_crawler.fetch_safe(str(item.get("url", "")))
                    for item in candidates
                ),
                return_exceptions=True,
            )
            for item, page in zip(candidates, pages):
                if isinstance(page, Exception) or page is None:
                    failed_count += 1
                    continue
                content = str(page.get("content", "")).strip()
                if content:
                    item["content"] = content[:8000]
                    enriched_count += 1
                else:
                    failed_count += 1
        logger.info("Agent web_search result_count=%s", len(results))
        await emit(
            {
                "type": "status",
                "stage": "web_search",
                "source_count": len(results),
                "message": (
                    (
                        f"联网搜索到 {len(results)} 条结果，"
                        f"已读取 {enriched_count} 个网页正文"
                        + (
                            f"，{failed_count} 个页面读取失败"
                            if failed_count
                            else ""
                        )
                    )
                    if results
                    else "联网搜索未找到相关结果"
                ),
                "query": search_query,
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
                f"{item.get('content') or item.get('snippet', '')}"
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
