from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from app.agent.graph import AssistantGraph
from app.services.memory import LongTermMemory
from app.storage.repositories import ChatRepository


class _Acquire:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self.connection

    async def __aexit__(self, *_: Any) -> None:
        return None


class _FakeConnection:
    def __init__(
        self,
        *,
        exists: bool = False,
        content: str | None = None,
        message_ids: list[dict[str, UUID]] | None = None,
        delete_result: str = "DELETE 1",
    ) -> None:
        self.exists = exists
        self.content = content
        self.message_ids = message_ids or []
        self.delete_result = delete_result

    async def fetchval(self, query: str, *_: Any) -> Any:
        if "SELECT EXISTS" in query:
            return self.exists
        if "SELECT m.content" in query:
            return self.content
        return None

    async def fetch(self, _: str, *__: Any) -> list[dict[str, UUID]]:
        return self.message_ids

    async def execute(self, *_: Any) -> str:
        return self.delete_result


class _FakeDatabase:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


class _FakeCache:
    def __init__(self) -> None:
        self.deleted_ids: list[UUID] = []
        self.deleted_messages: list[UUID] = []

    async def get_message_content(self, _: UUID) -> str:
        return "stale cached content"

    async def delete_message_content(self, message_id: UUID) -> None:
        self.deleted_messages.append(message_id)

    async def delete_session(
        self,
        _: UUID,
        *,
        message_ids: list[UUID] | None = None,
    ) -> None:
        self.deleted_ids = list(message_ids or [])


class _FakeRecall:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def recall_safe(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(str(kwargs["query"]))
        return []


class _FakeLLM:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(llm_model="test-model")

    async def create_plan(self, **_: Any) -> list[dict[str, str]]:
        return []

    async def select_tools(self, **_: Any) -> list[dict[str, object]]:
        return []

    async def stream_answer(self, **_: Any):
        yield "ok"


class _FakeRag:
    async def search_safe(self, _: str) -> list[dict[str, Any]]:
        return []


class _DeleteRepository:
    def __init__(self) -> None:
        self.called = False

    async def soft_delete(self, **_: Any) -> bool:
        self.called = True
        return True


class _BrokenVectorStore:
    def __init__(self) -> None:
        self.called = False

    async def soft_delete(self, _: UUID) -> None:
        self.called = True
        raise RuntimeError("qdrant unavailable")


class _StaleVectorStore:
    def __init__(self, fact_id: UUID) -> None:
        self.fact_id = fact_id

    async def search_active(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": str(self.fact_id),
                "score": 0.99,
                "payload": {
                    "fact_id": str(self.fact_id),
                    "content": "should not be used",
                },
            }
        ]


class _Embeddings:
    async def embed(self, _: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


class _EmptyActiveRepository:
    async def active_by_ids(
        self,
        *_: Any,
        **__: Any,
    ) -> dict[UUID, Any]:
        return {}

    async def touch(self, *_: Any, **__: Any) -> None:
        return None

    async def list_recent_deleted(self, **_: Any) -> list[Any]:
        return []


class MemoryConsistencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_previous_session_recall_requires_explicit_request(self) -> None:
        recall = _FakeRecall()
        agent = AssistantGraph(
            _FakeLLM(),
            _FakeRag(),
            conversation_recall=recall,
        )

        async def emit(_: dict[str, Any]) -> None:
            return None

        config = {"configurable": {"emit": emit}}
        base_state = {
            "session_id": str(uuid4()),
            "tenant_id": "tenant",
            "user_id": "user",
            "client_id": "client",
        }
        await agent._load_memory(
            {**base_state, "question": "今天天气怎么样"},
            config,
        )
        self.assertEqual(recall.calls, [])

        await agent._load_memory(
            {**base_state, "question": "上次我们聊到哪了"},
            config,
        )
        self.assertEqual(recall.calls, ["上次我们聊到哪了"])

    async def test_memory_delete_succeeds_when_vector_cleanup_fails(self) -> None:
        repository = _DeleteRepository()
        vectors = _BrokenVectorStore()
        memory = LongTermMemory(
            SimpleNamespace(memory_enabled=True),
            repository,
            None,
            vectors,
            None,
        )

        deleted = await memory.delete("tenant", "user", uuid4())

        self.assertTrue(deleted)
        self.assertTrue(repository.called)
        self.assertTrue(vectors.called)

    async def test_stale_vector_cannot_resurrect_deleted_memory(self) -> None:
        fact_id = uuid4()
        settings = SimpleNamespace(
            memory_enabled=True,
            embeddings_configured=True,
            memory_top_k=5,
            memory_retrieval_threshold=0.35,
            memory_recency_days=90,
        )
        memory = LongTermMemory(
            settings,
            _EmptyActiveRepository(),
            _Embeddings(),
            _StaleVectorStore(fact_id),
            None,
        )

        active, deleted = await memory.retrieve_safe("tenant", "user", "query")

        self.assertEqual(active, [])
        self.assertEqual(deleted, [])

    async def test_stale_message_cache_is_not_served_after_database_miss(
        self,
    ) -> None:
        cache = _FakeCache()
        repository = ChatRepository(
            _FakeDatabase(_FakeConnection(exists=False, content=None)),
            cache,
        )
        message_id = uuid4()

        content = await repository.get_message_content(
            message_id,
            client_id="client",
            tenant_id="tenant",
            user_id="user",
        )

        self.assertIsNone(content)
        self.assertEqual(cache.deleted_messages, [message_id])

    async def test_session_delete_purges_all_message_cache_keys(self) -> None:
        first_id = uuid4()
        second_id = uuid4()
        cache = _FakeCache()
        repository = ChatRepository(
            _FakeDatabase(
                _FakeConnection(
                    message_ids=[{"id": first_id}, {"id": second_id}],
                )
            ),
            cache,
        )
        session_id = uuid4()

        deleted = await repository.delete_session(
            session_id,
            client_id="client",
            tenant_id="tenant",
            user_id="user",
        )

        self.assertTrue(deleted)
        self.assertEqual(cache.deleted_ids, [first_id, second_id])


if __name__ == "__main__":
    unittest.main()
