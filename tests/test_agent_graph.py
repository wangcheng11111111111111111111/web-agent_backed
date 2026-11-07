from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from app.agent.graph import AssistantGraph
from app.config import DEFAULT_SYSTEM_PROMPT


class FakeLLM:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(llm_model="test-model")
        self.answer_tool_context = ""
        self.answer_plan_context = ""

    async def create_plan(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        available_tools: list[str],
    ) -> list[dict[str, str]]:
        del message, history, available_tools
        return [
            {"title": "确认计算目标", "tool": ""},
            {"title": "执行精确计算", "tool": "calculate"},
            {"title": "检索补充资料", "tool": "search_knowledge_base"},
            {"title": "整合结果并核对", "tool": ""},
        ]

    async def select_tools(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        tools: list[dict[str, object]],
        plan: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        del message, history, tools, plan
        return [
            {"name": "calculate", "arguments": {"expression": "2 + 3"}},
            {"name": "search_knowledge_base", "arguments": {}},
        ]

    async def stream_answer(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        knowledge_context: str,
        tool_context: str,
        long_term_context: str,
        conversation_context: str,
        execution_plan: str,
    ):
        del message, history, knowledge_context, long_term_context
        del conversation_context
        self.answer_tool_context = tool_context
        self.answer_plan_context = execution_plan
        yield "计算结果是 5；知识库检索失败，"
        yield "因此补充资料部分未能完成。"


class FailingRag:
    async def search_safe(self, question: str) -> list[dict[str, Any]]:
        del question
        raise RuntimeError("index unavailable")


class AgentPlanningTests(unittest.IsolatedAsyncioTestCase):
    def test_system_prompt_prioritizes_fresh_tool_results(self) -> None:
        self.assertIn("本轮工具结果是当前有效信息", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("忽略历史", DEFAULT_SYSTEM_PROMPT)

    def test_recent_is_treated_as_realtime(self) -> None:
        self.assertTrue(AssistantGraph._looks_realtime("最近的major冠军"))
        self.assertTrue(AssistantGraph._looks_realtime("recent major winner"))

    async def test_tool_exception_is_isolated_and_reported(self) -> None:
        llm = FakeLLM()
        agent = AssistantGraph(llm, FailingRag())
        events: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)

        result = await agent.run(
            question="计算 2 + 3，并查知识库补充",
            history=[],
            emit=emit,
        )

        self.assertIn("2 + 3 = 5", llm.answer_tool_context)
        self.assertIn("知识库检索执行异常", llm.answer_tool_context)
        self.assertIn("已跳过该步骤并继续回答", llm.answer_tool_context)
        self.assertIn("[completed] 执行精确计算", llm.answer_plan_context)
        self.assertIn("[failed] 检索补充资料", llm.answer_plan_context)
        self.assertIn("知识库检索失败", result["answer"])

        plan_events = [event for event in events if event["type"] == "plan"]
        self.assertEqual(len(plan_events), 1)
        self.assertEqual(len(plan_events[0]["steps"]), 4)
        self.assertEqual(result["execution_plan"][1]["status"], "completed")
        self.assertEqual(result["execution_plan"][2]["status"], "failed")
        self.assertTrue(
            any(
                event.get("type") == "plan_step"
                and event.get("status") == "failed"
                for event in events
            )
        )

    async def test_planning_failure_uses_fallback_and_answer_still_runs(self) -> None:
        class BrokenPlannerLLM(FakeLLM):
            async def create_plan(self, **_: Any) -> list[dict[str, str]]:
                raise TimeoutError("planner timeout")

            async def select_tools(self, **_: Any) -> list[dict[str, object]]:
                return []

        llm = BrokenPlannerLLM()
        agent = AssistantGraph(llm, FailingRag())

        async def emit(event: dict[str, Any]) -> None:
            del event

        result = await agent.run(question="你好", history=[], emit=emit)

        self.assertEqual(len(result["execution_plan"]), 3)
        self.assertIn("TimeoutError", result["plan_error"])
        self.assertTrue(result["answer"])
        self.assertEqual(result["execution_plan"][-1]["status"], "completed")


if __name__ == "__main__":
    asyncio.run(unittest.main())
