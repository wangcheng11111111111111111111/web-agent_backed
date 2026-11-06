from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID

from app.config import Settings
from app.storage.repositories import ChatRepository

logger = logging.getLogger(__name__)

_STOP_TERMS = {
    "什么",
    "怎么",
    "如何",
    "可以",
    "这个",
    "那个",
    "以及",
    "一下",
    "帮我",
    "问题",
}


def _tokens(text: str) -> set[str]:
    lowered = text.casefold()
    tokens = {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_.-]{1,}", lowered)
        if word not in {"the", "and", "for", "with"}
    }
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        if segment not in _STOP_TERMS:
            tokens.add(segment)
        if len(segment) > 2:
            tokens.update(
                segment[index : index + 2]
                for index in range(len(segment) - 1)
            )
    return tokens - _STOP_TERMS


class ConversationRecall:
    """Fallback retrieval from a user's previous sessions.

    This is intentionally lightweight and local: PostgreSQL reads a bounded
    recent window, then bilingual lexical overlap plus recency picks context.
    It is used when no long-term facts are available, or when the user asks to
    continue an earlier topic.
    """

    def __init__(
        self,
        settings: Settings,
        chat_repository: ChatRepository,
    ) -> None:
        self.settings = settings
        self.chat_repository = chat_repository

    async def recall_safe(
        self,
        *,
        client_id: str,
        tenant_id: str,
        user_id: str,
        current_session_id: UUID,
        query: str,
    ) -> list[dict[str, Any]]:
        if not self.settings.conversation_recall_enabled:
            return []
        try:
            candidates = await self.chat_repository.recall_recent_messages(
                client_id=client_id,
                tenant_id=tenant_id,
                user_id=user_id,
                exclude_session_id=current_session_id,
                limit=self.settings.conversation_recall_scan,
            )
            if not candidates:
                return []
            query_tokens = _tokens(query)
            scored: list[tuple[float, dict[str, Any]]] = []
            for index, message in enumerate(candidates):
                message_tokens = _tokens(str(message["content"]))
                overlap = len(query_tokens & message_tokens)
                recency = index / max(len(candidates), 1)
                score = overlap * 10 + recency
                scored.append((score, message))
            best = sorted(scored, key=lambda item: item[0], reverse=True)
            positive = [item for item in best if item[0] >= 10]
            chosen = positive[: self.settings.conversation_recall_limit]
            if not chosen:
                chosen = sorted(
                    scored[-self.settings.conversation_recall_limit :],
                    key=lambda item: item[1]["seq"],
                )
            chosen.sort(key=lambda item: item[1]["seq"])

            results: list[dict[str, Any]] = []
            used_chars = 0
            for score, message in chosen:
                content = str(message["content"]).strip()
                if not content:
                    continue
                remaining = (
                    self.settings.conversation_recall_max_chars - used_chars
                )
                if remaining <= 0:
                    break
                content = content[:remaining]
                used_chars += len(content)
                results.append(
                    {
                        "message_id": str(message["id"]),
                        "session_id": str(message["session_id"]),
                        "session_title": str(message["session_title"]),
                        "role": str(message["role"]),
                        "content": content,
                        "created_at": message["created_at"].isoformat(),
                        "score": round(score, 3),
                    }
                )
            logger.info(
                "Conversation recall tenant_id=%s user_id=%s messages=%s",
                tenant_id,
                user_id,
                len(results),
            )
            return results
        except Exception as exc:
            logger.warning(
                "Conversation recall failed safely (%s)",
                type(exc).__name__,
            )
            return []