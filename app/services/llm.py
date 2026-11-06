from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

from app.config import Settings
from app.errors import (
    ModelConfigError,
    ModelTimeoutError,
    ModelUpstreamError,
)

if TYPE_CHECKING:
    from app.storage.prompt_store import PromptStore

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings, prompt_store: PromptStore | None = None) -> None:
        self.settings = settings
        self._prompt_store = prompt_store

    async def _messages(
        self,
        message: str,
        history: list[dict[str, str]],
        knowledge_context: str,
        tool_context: str,
        long_term_context: str,
        conversation_context: str,
    ) -> list[dict[str, str]]:
        if self._prompt_store is not None:
            system = await self._prompt_store.get_full_system_prompt()
        else:
            from app.config import DEFAULT_SYSTEM_PROMPT
            system = DEFAULT_SYSTEM_PROMPT
        if knowledge_context:
            system += "\n\n知识库片段：\n" + knowledge_context
        if tool_context:
            system += "\n\n工具执行结果：\n" + tool_context
        if long_term_context:
            system += "\n\n用户长期记忆：\n" + long_term_context
        if conversation_context:
            system += (
                "\n\n往期会话参考（这些是历史对话原文，不是长期事实；"
                "只在相关时参考，若与当前消息冲突以当前消息为准）：\n"
                + conversation_context
            )
        messages = [{"role": "system", "content": system}]
        for item in history[-20:]:
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    async def stream_answer(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        knowledge_context: str = "",
        tool_context: str = "",
        long_term_context: str = "",
        conversation_context: str = "",
    ) -> AsyncIterator[str]:
        if not self.settings.model_configured:
            raise ModelConfigError()

        payload = {
            "model": self.settings.llm_model,
            "messages": await self._messages(
                message,
                history,
                knowledge_context,
                tool_context,
                long_term_context,
                conversation_context,
            ),
            "stream": True,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        timeout = httpx.Timeout(
            self.settings.llm_timeout,
            connect=min(15.0, self.settings.llm_timeout),
        )
        url = f"{self.settings.llm_base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    if response.status_code >= 400:
                        logger.error(
                            "LLM upstream returned HTTP %s",
                            response.status_code,
                        )
                        raise ModelUpstreamError()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            choice = (event.get("choices") or [{}])[0]
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError, TypeError):
                            logger.warning("Ignoring malformed LLM stream chunk")
        except (ModelConfigError, ModelUpstreamError):
            raise
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError() from exc
        except httpx.HTTPError as exc:
            logger.error("LLM request failed: %s", type(exc).__name__)
            raise ModelUpstreamError() from exc

    async def extract_memories(
        self,
        *,
        user_message: str,
        assistant_answer: str,
    ) -> list[dict[str, object]]:
        """Extract candidate stable user facts with a structured model call."""
        if not self.settings.model_configured:
            logger.info("Long-term memory extraction skipped: model config missing")
            return []
        system = (
            "你是用户长期记忆提取器。只提取用户在消息中明确陈述、"
            "未来对话仍可能有用的事实。偏好、习惯、技能、关系、近期计划和"
            "阶段性上下文都可以保留，不必等到永久不变。"
            "不要从助手回答、寒暄、临时问题或推测中提取。"
            "不要提取密码、密钥、验证码、身份证、银行卡等敏感信息。"
            "返回严格 JSON，不要 Markdown。格式："
            '{"facts":[{"fact_type":"profile|preference|goal|constraint|project|'
            'habit|skill|relationship|context",'
            '"key":"稳定字段名，如回答风格/当前项目","value":"简洁的独立陈述句",'
            '"subject":"事实主体","confidence":0.0到1.0,'
            '"expires_in_days":null或正整数}]}。'
            "同一事实必须使用相同 key，便于后续更新而不是重复保存。"
            "没有值得保存的事实时返回 {\"facts\":[]}。"
        )
        prompt = (
            "【用户消息】\n"
            f"{user_message}\n\n"
            "【助手回答，仅作上下文，不可作为提取依据】\n"
            f"{assistant_answer[:4000]}"
        )
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(
            min(self.settings.llm_timeout, 30.0),
            connect=min(10.0, self.settings.llm_timeout),
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.settings.llm_base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.settings.llm_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                    },
                )
            if response.status_code >= 400:
                logger.warning(
                    "Memory extraction upstream returned HTTP %s",
                    response.status_code,
                )
                return []
            payload = response.json()
            content = (
                ((payload.get("choices") or [{}])[0].get("message") or {})
                .get("content", "")
            )
            if not isinstance(content, str) or not content.strip():
                return []
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].lstrip()
            parsed = json.loads(content)
            facts = parsed.get("facts", [])
            return facts[:6] if isinstance(facts, list) else []
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Memory extraction failed safely (%s)",
                type(exc).__name__,
            )
            return []
