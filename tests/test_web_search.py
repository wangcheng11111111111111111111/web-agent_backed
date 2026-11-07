from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import httpx

from app.agent.graph import AssistantGraph
from app.services.web_search import WebSearchError, WebSearchService


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        web_search_enabled=True,
        web_search_provider="tavily",
        web_search_api_key="test-key",
        web_search_top_k=5,
        web_search_timeout=10,
    )


class _FailingWebSearch:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            web_search_enabled=True,
            web_search_provider="tavily",
        )

    async def search_safe(self, query: str) -> list[dict[str, Any]]:
        del query
        raise WebSearchError("Tavily 联网搜索失败（ConnectError）")


class _WebPlanLLM:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(llm_model="test-model")
        self.tool_context = ""

    async def create_plan(self, **_: Any) -> list[dict[str, str]]:
        return [
            {"title": "理解问题", "tool": ""},
            {"title": "联网检索", "tool": "web_search"},
            {"title": "组织回答", "tool": ""},
        ]

    async def select_tools(self, **_: Any) -> list[dict[str, object]]:
        return [{"name": "web_search", "arguments": {"query": "latest news"}}]

    async def stream_answer(self, **kwargs: Any):
        self.tool_context = str(kwargs["tool_context"])
        yield "联网搜索暂时不可用，我将基于现有信息继续回答。"


class _EmptyRag:
    async def search_safe(self, _: str) -> list[dict[str, Any]]:
        return []


class WebSearchFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_failure_does_not_fall_back_to_bing(self) -> None:
        service = WebSearchService(_settings())
        bing_called = False

        async def fail_tavily(_: str, __: str) -> list[dict[str, Any]]:
            request = httpx.Request("POST", "https://api.tavily.com/search")
            raise httpx.ConnectError("offline", request=request)

        async def fake_bing(_: str) -> list[dict[str, Any]]:
            nonlocal bing_called
            bing_called = True
            return [{"title": "fallback", "url": "https://example.com"}]

        service._tavily = fail_tavily  # type: ignore[method-assign]
        service._search_bing = fake_bing  # type: ignore[method-assign]

        with self.assertRaisesRegex(WebSearchError, "Tavily.*ConnectError"):
            await service.search("latest news")
        self.assertFalse(bing_called)

    async def test_unknown_provider_is_configuration_error(self) -> None:
        settings = _settings()
        settings.web_search_provider = "unsupported"
        service = WebSearchService(settings)

        with self.assertRaisesRegex(WebSearchError, "配置无效"):
            await service.search("latest news")

    async def test_agent_reports_tool_failure_and_continues(self) -> None:
        llm = _WebPlanLLM()
        agent = AssistantGraph(
            llm,
            _EmptyRag(),
            web_search=_FailingWebSearch(),
        )
        events: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)

        result = await agent.run(
            question="搜索最新新闻",
            history=[],
            emit=emit,
            enable_web_search=True,
        )

        self.assertIn("Tavily 联网搜索失败", result["tool_error"])
        self.assertIn("已跳过联网搜索并继续回答", result["answer"])
        self.assertEqual(result["execution_plan"][1]["status"], "failed")
        self.assertTrue(
            any(
                event.get("type") == "status"
                and event.get("stage") == "web_search"
                and event.get("error") is True
                for event in events
            )
        )


if __name__ == "__main__":
    unittest.main()
