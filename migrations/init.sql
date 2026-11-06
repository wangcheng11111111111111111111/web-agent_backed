-- ============================================================
-- sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    title VARCHAR(120) NOT NULL,
    client_id VARCHAR(64) NOT NULL DEFAULT 'default',
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

UPDATE sessions
SET user_id = client_id
WHERE user_id = 'default' AND client_id <> 'default';

CREATE INDEX IF NOT EXISTS idx_sessions_client_updated
    ON sessions(client_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_tenant_user_updated
    ON sessions(tenant_id, user_id, updated_at DESC);

-- ============================================================
-- messages
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS seq BIGSERIAL;

CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(session_id, seq DESC)
    INCLUDE (id, role, created_at);

-- ============================================================
-- documents
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(120) NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
    ON document_chunks(document_id, chunk_index);

-- ============================================================
-- user_facts (长期记忆)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_facts (
    id UUID PRIMARY KEY,
    client_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id VARCHAR(64) NOT NULL DEFAULT 'default',
    fact_type VARCHAR(32) NOT NULL,
    subject VARCHAR(120) NOT NULL,
    memory_key VARCHAR(120) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    source_session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    deleted_reason VARCHAR(32),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE user_facts
    DROP CONSTRAINT IF EXISTS user_facts_fact_type_check;

ALTER TABLE user_facts
    ADD CONSTRAINT user_facts_fact_type_check CHECK (
        fact_type IN (
            'profile',
            'preference',
            'goal',
            'constraint',
            'project',
            'habit',
            'skill',
            'relationship',
            'context'
        )
    );

-- 回填：user_id 从 client_id 迁移
UPDATE user_facts
SET user_id = client_id
WHERE user_id = 'default' AND client_id <> 'default';

-- 回填：memory_key
UPDATE user_facts
SET memory_key = fact_type || ':' || lower(subject)
WHERE memory_key IS NULL OR memory_key = '';

-- 回填：content_hash
UPDATE user_facts
SET content_hash = md5(
    lower(regexp_replace(btrim(content), '\s+', ' ', 'g'))
)
WHERE content_hash IS NULL;

-- 回填：importance
UPDATE user_facts
SET importance = CASE fact_type
    WHEN 'profile' THEN 0.85
    WHEN 'constraint' THEN 0.80
    WHEN 'goal' THEN 0.75
    WHEN 'project' THEN 0.70
    WHEN 'preference' THEN 0.65
    ELSE 0.50
END
WHERE importance = 0.5;

-- 去重：相同 content_hash 只保留最新一条
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY tenant_id, user_id, content_hash
               ORDER BY updated_at DESC, created_at DESC, id DESC
           ) AS duplicate_rank
    FROM user_facts
    WHERE is_active = TRUE
)
UPDATE user_facts AS fact
SET is_active = FALSE,
    deleted_reason = 'deduplicated',
    updated_at = NOW()
FROM ranked
WHERE fact.id = ranked.id
  AND ranked.duplicate_rank > 1;

DROP INDEX IF EXISTS uq_user_facts_active_hash;

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_facts_active_identity
    ON user_facts(tenant_id, user_id, content_hash)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_user_facts_active_lookup
    ON user_facts(tenant_id, user_id, fact_type, memory_key, updated_at DESC)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_user_facts_deleted_recent
    ON user_facts(tenant_id, user_id, updated_at DESC)
    WHERE is_active = FALSE;

-- ============================================================
-- app_settings
-- ============================================================
CREATE TABLE IF NOT EXISTS app_settings (
    key VARCHAR(64) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- client_branding（从 app_settings 迁移品牌配置）
-- ============================================================
CREATE TABLE IF NOT EXISTS client_branding (
    client_id VARCHAR(64) NOT NULL,
    setting_key VARCHAR(64) NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (client_id, setting_key)
);

CREATE INDEX IF NOT EXISTS idx_client_branding_updated
    ON client_branding(client_id, updated_at DESC);

INSERT INTO client_branding (client_id, setting_key, value, updated_at)
SELECT DISTINCT
    sessions.client_id,
    settings.key,
    settings.value,
    settings.updated_at
FROM app_settings AS settings
CROSS JOIN sessions
WHERE settings.key IN (
    'assistant_name',
    'assistant_logo_file',
    'assistant_background_enabled',
    'assistant_background_type',
    'assistant_background_video'
)
ON CONFLICT (client_id, setting_key) DO NOTHING;

DELETE FROM app_settings
WHERE key IN (
    'assistant_name',
    'assistant_logo_file',
    'assistant_background_enabled',
    'assistant_background_type',
    'assistant_background_video'
);

-- ============================================================
-- app_assets（图片/视频等二进制资产）
-- ============================================================
CREATE TABLE IF NOT EXISTS app_assets (
    name VARCHAR(64) PRIMARY KEY,
    content_type VARCHAR(64) NOT NULL,
    data BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);