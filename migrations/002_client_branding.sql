CREATE TABLE IF NOT EXISTS client_branding (
    client_id VARCHAR(64) NOT NULL,
    setting_key VARCHAR(64) NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (client_id, setting_key)
);

CREATE INDEX IF NOT EXISTS idx_client_branding_updated
    ON client_branding(client_id, updated_at DESC);

-- Preserve the previous global branding for existing clients exactly once.
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
