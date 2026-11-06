from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.services.memory import MemoryVectorStore
from app.services.rag import QdrantStore


async def run() -> None:
    settings = get_settings()
    store = QdrantStore(settings)
    await store.ensure_collection()
    memory_store = MemoryVectorStore(settings)
    await memory_store.ensure_collection()
    migrated = await memory_store.backfill_legacy_scope("default-tenant")
    print(
        "Qdrant 集合已就绪："
        f"{settings.qdrant_collection} / {settings.memory_collection}，"
        f"维度 {settings.embedding_dim}，回填旧记忆 {migrated} 条"
    )


if __name__ == "__main__":
    asyncio.run(run())