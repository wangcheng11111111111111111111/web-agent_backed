from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg

from app.config import get_settings


async def run() -> None:
    settings = get_settings()
    migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
    connection = await asyncpg.connect(dsn=settings.database_dsn)
    try:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(128) PRIMARY KEY,
                checksum VARCHAR(64) NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await connection.execute("SELECT pg_advisory_lock(4815162342)")
        applied = {
            row["version"]: row["checksum"]
            for row in await connection.fetch(
                "SELECT version, checksum FROM schema_migrations"
            )
        }
        for migration_path in sorted(migrations_dir.glob("*.sql")):
            version = migration_path.name
            content = migration_path.read_text(encoding="utf-8-sig")
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            previous_checksum = applied.get(version)
            if previous_checksum is not None:
                if previous_checksum != checksum:
                    raise RuntimeError(
                        f"Migration {version} changed after it was applied"
                    )
                print(f"Skipped {version}")
                continue
            async with connection.transaction():
                await connection.execute(content)
                await connection.execute(
                    """
                    INSERT INTO schema_migrations (version, checksum)
                    VALUES ($1, $2)
                    """,
                    version,
                    checksum,
                )
            print(f"Applied {version}")
    finally:
        await connection.execute("SELECT pg_advisory_unlock(4815162342)")
        await connection.close()
    print("数据库迁移完成")


if __name__ == "__main__":
    asyncio.run(run())