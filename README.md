# 后端

本仓库包含 FastAPI 后端、CosyVoice2/RVC 运行时代码、数据库迁移、Docker Compose
编排和模型准备脚本。

前端是独立仓库：

```text
https://github.com/wangcheng11111111111111111111/web-agent_fronted.git
```

完整部署时建议将两个仓库检出为同级目录：

```text
workspace/
├── backend/
└── frontend/
```

## 技术结构

- FastAPI：健康检查、聊天、SSE、会话与知识库 API。
- LangGraph：`classify -> load_memory -> time_tool/retrieve -> answer` 条件图。
- httpx：DeepSeek/OpenAI 兼容模型和 Embedding HTTP 调用。
- Redis：最近 40 条会话消息热缓存，TTL 默认 7 天。
- PostgreSQL：会话、消息、文档元数据和切片。
- Qdrant：知识库和用户长期事实的 1024 维向量索引。

## 启动

### 本地开发

```powershell
git clone https://github.com/wangcheng11111111111111111111/web-agent_backed.git
cd web-agent_backed
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# 编辑 .env 后：
python scripts/migrate.py
python scripts/init_qdrant.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

这个仓库不包含 `.env`、`.venv/`、`models/`、`data/` 或 RVC 通用资源。
首次运行前必须根据 `.env.example` 配置环境，并放置所需模型资源。
依赖安装后可执行：

```powershell
python scripts/download_models.py
```

该脚本下载 CosyVoice2、HuBERT 和 RMVPE；用户自己的 RVC `.pth` 和 `.index`
不会被自动下载，需要放入 `models/rvc/weights/` 与 `models/rvc/indexes/`。
完整清单和目录树见 [MODELS.md](MODELS.md)。

也可以在 PyCharm 中直接运行 `app/main.py`，或执行：

```powershell
python app/main.py
```

直接运行会读取 `.env` 的 `API_HOST`/`API_PORT` 并启动 Uvicorn。修改代码时如需
自动重载，请使用上面的 `--reload` 命令。

### Docker 一键启动

先将前端仓库检出为本仓库同级的 `frontend/` 目录，填写 `.env` 后执行：

```powershell
.\scripts\docker-up.ps1
```

Linux/macOS 使用：

```bash
sh scripts/docker-up.sh
```

Compose 会启动 PostgreSQL、Redis、Qdrant、数据库迁移、GPU 后端和 Nginx 前端，
默认访问地址为 `http://127.0.0.1:8080`。GPU、模型目录、端口与排障说明见
[DOCKER.md](DOCKER.md)。

`GET /api/health` 不依赖模型、Redis、PostgreSQL 或 Qdrant。聊天与历史会话需要
PostgreSQL；Redis 不可用时自动回源 PostgreSQL；知识库上传和语义检索需要
Embedding 密钥及 Qdrant。

## 本地朗读

CosyVoice2 需要已下载的模型和 CUDA 版 PyTorch。RVC 本地音色还需要以下公开
推理资源，目录结构必须保持一致：

```text
vendor/RVC/assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
└── rmvpe/rmvpe.pt
```

资源来自 Hugging Face 仓库 `lj1995/VoiceConversionWebUI`。RVC 音频处理使用
`imageio-ffmpeg` 自带的 FFmpeg，不要求系统 PATH 中额外安装 `ffmpeg`。
用户自己的 `.pth` 模型和 `.index` 索引分别放在
`models/rvc/weights/` 与 `models/rvc/indexes/`。

CosyVoice2 的零样本语气由参考音频和对应文本共同决定。前端可以上传多个语气片段、
试听、重命名和重新识别；上传时也可以手动填写参考文本。自动识别优先使用：

```ini
STT_ENABLED=true
STT_DEVICE=auto
STT_WHISPER_MODEL=small
STT_BASE_URL=https://api.siliconflow.cn/v1
STT_API_KEY=你的语音识别密钥
STT_MODEL=FunAudioLLM/SenseVoiceSmall
```

远程 STT 未配置或失败时会回退到本地 Whisper Small，并优先使用 CUDA。识别仍失败时音频不会丢弃，
`text_source` 会标记为 `missing`，用户可以稍后手动补全。内置默认语气不会被删除。
参考音频建议使用 3–15 秒并不得超过 30 秒，否则 CosyVoice2 的提示编码和合成会
明显变慢。

语气预设保存在 `data/voice_prompts/`：

```text
data/voice_prompts/
├── prompts.json
└── <uuid>.wav
```

RVC 的 `index_rate`、`protect`、`rms_mix_rate` 和 `f0_up_key` 可通过流式 TTS
请求传入，并在前端设置中持久化。它们只影响音色转换，不负责生成情绪和韵律。

语音接口只保留：

```text
POST /api/audio/tts/stream
```

返回 `application/x-ndjson` PCM 分片，由前端 Web Audio 连续播放。

语气预设接口：

```text
GET    /api/audio/prompts
POST   /api/audio/prompts
GET    /api/audio/prompts/{id}/audio
PUT    /api/audio/prompts/{id}
POST   /api/audio/prompts/{id}/transcribe
DELETE /api/audio/prompts/{id}
```

## 测试

```powershell
python -m compileall -q app
```

如果数据库密码包含 `@`、`:`、`/` 等字符，无需手动转义；后端会安全编码 DSN。

## LangGraph

```text
START -> classify -> load_memory
                         |-- 时间关键词 -> time_tool -> answer -> END
                         |-- 其他问题   -> retrieve  -> answer -> END
```

- `classify`：识别“现在几点、今天日期、what time”等时间意图。
- `load_memory`：按 `X-Tenant-ID` + `X-User-ID` 检索长期事实，注入 answer 上下文。
  当长期记忆为空，或问题包含“上次/之前/继续/记得”等词时，会从同一用户的
  其他会话回退检索相关历史消息。
- `time_tool`：真实读取后端当前 UTC+8 时间；异常时降级为普通回答。
- `retrieve`：调用 Embedding 与 Qdrant；配置缺失、空库或检索失败时继续普通聊天。
- `answer`：通过 httpx 调用 DeepSeek 流式接口，并逐块发出事件。

回答完成并保存消息后，后台 `memory_chain` 执行：

1. DeepSeek 结构化提取 `type/key/value/subject/confidence/expires_in_days`。
2. Python 校验类型、Key、长度、置信度、敏感词、疑问句和过期上限。
3. 按 tenant/user 检索相似记忆；明确纠正时只淘汰相关旧事实。
4. PostgreSQL 用内容哈希做精确去重，Qdrant 做语义去重和召回。
5. PostgreSQL 写入事实和访问统计，Qdrant 写入向量及过滤字段。

日志会记录 `node`、`route`、工具执行结果、RAG/记忆数量与错误类型，不记录
模型密钥或完整敏感内容。

## 存储优化

- Redis 使用 List 保存最近 40 条消息，`RPUSH + LTRIM + EXPIRE` 原子更新，TTL 默认 7 天。
- Redis 不可用时自动回源 PostgreSQL；旧版整段 JSON 缓存会自动迁移为 List。
- PostgreSQL 使用 `schema_migrations` 记录版本和 SHA-256 校验和，避免重复执行或静默修改迁移。
- `user_facts` 以 `(tenant_id, user_id, content_hash)` 建立活跃事实唯一约束，并记录重要度、访问次数、最后访问时间和过期时间。
- Qdrant 为 `tenant_id`、`user_id`、`fact_type`、`is_active` 建立 payload 索引。
- 助手名称、Logo 文件名和背景状态存于 `app_settings`；图片写入
  `data/branding/`，视频写入 `data/backgrounds/`。
- 图片格式支持 PNG、JPEG、WebP、GIF。
- 视频上传按 1 MB 分块流式写盘，最大 10 GB，不会整体载入 Python 内存。
- 后端媒体库列出已上传图片和视频，支持切换图标/背景用途及手动删除。
- 上传图片后可在同一设置页一键设为页面背景；背景状态存于
  `assistant_background_enabled`，正文区域自动叠加深色遮罩保证可读性。

## 往期会话回退

长期记忆为空时，后端从 PostgreSQL 读取当前 tenant/user 最近
`CONVERSATION_RECALL_SCAN` 条历史消息，排除当前会话，再按中英文关键词重合度
和时间排序，最多注入 `CONVERSATION_RECALL_LIMIT` 条、共
`CONVERSATION_RECALL_MAX_CHARS` 字符。长期记忆存在时，用户明确提到“上次、
之前、继续、记得”也会补充往期对话。
