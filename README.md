# Nebula AI Assistant 项目文档

> 对应当前后端代码，最后更新于 2026-11-06。

## 1. 项目概览

Nebula 是一个本地优先的 AI 对话与语音助手，由两个独立仓库组成：

```text
workspace/
├── backend/      FastAPI、LangGraph、PostgreSQL、Redis、Qdrant、TTS/RVC
└── frontend/     Vue 3、Vite、TypeScript、Web Audio
```

主要能力：

- OpenAI 兼容流式聊天。
- LangGraph 记忆加载、模型自主工具选择和工具执行。
- PostgreSQL 保存会话、消息、文档和长期事实。
- Redis 缓存最近会话及按消息 ID 查询的 TTS 正文。
- Qdrant 保存知识库和长期记忆向量。
- CosyVoice2、Edge TTS、RVC 音色转换。
- 后端逐段生成 PCM，前端连续调度播放。

## 2. 当前部署假设

当前为**单用户共享模式**：

```ini
GLOBAL_CLIENT_ID=...
GLOBAL_TENANT_ID=...
GLOBAL_USER_ID=...
```

所有请求统一使用这些配置，浏览器不能通过请求头切换身份。所有访问者共享同一组会话、记忆、知识库和品牌配置。

这不是正式认证。当前版本不要直接暴露到不可信公网。多用户改造前必须先实现登录、鉴权和数据归属校验。

## 3. 系统架构

```text
Browser / Vue + Web Audio
  ├── POST /api/chat/stream       SSE 文本流
  └── POST /api/audio/tts/stream  NDJSON 音频流

FastAPI
  ├── LangGraph
  │   └── select_tools -> load_memory -> execute_tools -> answer
  ├── PostgreSQL
  │   └── sessions/messages/documents/user_facts/settings
  ├── Redis
  │   └── recent messages/message content
  ├── Qdrant
  │   └── knowledge vectors/memory vectors
  ├── CosyVoice2 / Edge TTS
  └── RVC
```

外部服务：

- OpenAI 兼容聊天模型。
- OpenAI 兼容 Embedding 服务。
- Web Search 和 Web Crawler。
- 可选远程 STT 或本地 Whisper。

## 4. 核心业务流

### 4.1 聊天

1. 前端调用 `POST /api/chat/stream`。
2. 后端创建或校验会话并写入用户消息。
3. 对话模型通过原生 `tools/tool_calls` 自主选择时间、知识库、Bing
   搜索或网页抓取工具。
4. LangGraph 先加载长期记忆，再执行模型选择的工具。
5. SSE 持续返回状态和增量文本。
6. 完成后助手消息写入 PostgreSQL。
7. 完成事件返回真实数据库消息 UUID。
8. 前端替换临时 `local-assistant-*` ID。
9. 后台异步提取长期记忆。

模型工具调用使用 OpenAI 兼容 `tools/tool_calls`。可用工具包括当前时间、
安全算术计算、本地知识库检索、知识库文档列表、长期记忆列表、历史会话
检索、网页抓取和 Bing 搜索。模型为 `web_search` 生成包含对象、年份和
日期的查询词；后端在“今天/今日”查询中自动补充当天日期，先解析 Bing
HTML，无结果时改用 Bing RSS，并抓取前两条搜索结果正文辅助回答。搜索
不依赖 API Key，也不会自动切换其他搜索引擎。
CS2/CSGO 查询会自动识别今天、昨天、明天及显式日期，追加 HLTV 和
Liquipedia 站内查询，并优先抓取 HLTV 对应日期的结果页。
如果搜索结果页直连返回 403、反爬挑战或没有正文，后端会通过
`r.jina.ai` 纯文本阅读器再次读取页面；该兜底会把目标 URL 发送给第三方
阅读服务，不包含用户聊天正文或密钥。
如果模型漏选工具但问题明显属于今天、最新、比赛、新闻、天气、股价等
实时信息，后端会在联网开启时强制补调 Bing；联网关闭时会向回答上下文
明确说明无法获取实时数据。前端会记住最近一次联网开关状态。

### 4.2 TTS 取文

前端不再提交回答正文，只提交：

```json
{
  "message_id": "助手消息 UUID",
  "engine": "cosyvoice",
  "rvc_enabled": true,
  "rvc_model": "model-name",
  "rvc_index": "model.index"
}
```

后端读取顺序：

```text
message_id
  -> Redis: message:<id>:content
  -> Redis 未命中时查询 PostgreSQL
  -> 回填 Redis
  -> 清洗 Markdown
  -> 按段落和完整句子分组
  -> TTS
  -> 可选 RVC
  -> 每段立即返回
```

当前正文安全上限为 20,000 字符。旧版本曾限制 800 字符，可能造成长回答后半段未朗读。

### 4.3 分段策略

- 不同段落不合并。
- 单句最多约 64 字。
- 首段目标最多 120 字。
- 第二段最多 `min(80, 首段实际字数 - 1)`。
- 后续每段必须严格小于前一段的实际字数。
- 只有单个句子放不下时才在标点附近拆分。
- 不增加段间静音。
- 前端按上一段结束时间调度下一段。

### 4.4 TTS/RVC 流水线

```text
Task A: 生成底音 -> source_queue
Task B: 读取底音 -> RVC -> output_queue
HTTP: output_queue -> 立即发送前端
```

两个队列容量均为 1，最多提前处理一段。A/B 逻辑上并行，但 CosyVoice 和 RVC 共用 GPU 时，推理可能串行或互相争用。

### 4.5 引擎规则

- 请求开始前确定一个 TTS 引擎。
- 整条回答锁定同一引擎。
- 引擎失败直接报错。
- 不自动从 CosyVoice 切换 Edge TTS，也不反向切换。
- RVC 失败不静默换模型。

### 4.6 RVC 预加载

- 启动时预加载首个模型。
- 前端读取选择后预加载当前模型。
- 切换模型或启用 RVC 时预加载。
- 相同文件大小和修改时间不会重复预热。
- 接口：`POST /api/audio/rvc/preload?model_name=<name>`。

## 5. 目录结构

```text
backend/
├── app/
│   ├── agent/                 LangGraph
│   ├── services/              聊天、RAG、记忆、语音
│   ├── storage/               数据库和文件仓储
│   ├── config.py              严格环境变量配置
│   ├── database.py            异步 PostgreSQL 连接池
│   ├── errors.py              统一错误
│   ├── main.py                FastAPI 路由
│   └── schemas.py             API 模型
├── migrations/init.sql        PostgreSQL 结构
├── scripts/                   模型下载、迁移和启动脚本
├── models/                    不在 Git 中
│   ├── cosyvoice2/
│   └── rvc/{weights,indexes}/
├── vendor/RVC/assets/         不在 Git 中
├── data/                      运行时媒体
├── compose.yaml
└── requirements.txt
```

## 6. 环境要求

- Windows 10/11 + PowerShell
- Python 3.12
- Node.js 20.19+ 或 22.12+
- NVIDIA GPU，建议至少 10 GB 显存
- CUDA 12.8 对应 PyTorch
- Docker Desktop/Engine + Compose v2（推荐）
- 建议 32 GB 内存、20 GB 以上空闲磁盘

仅使用普通聊天和 Edge TTS 时不需要 CosyVoice/RVC GPU 模型。

## 7. 安装

### 7.1 克隆

```powershell
cd D:\projects
git clone <backend-url> backend
git clone <frontend-url> frontend
```

### 7.2 配置

```powershell
cd backend
Copy-Item .env.example .env
```

必须检查：

- `GLOBAL_CLIENT_ID`、`GLOBAL_TENANT_ID`、`GLOBAL_USER_ID`
- LLM 地址、模型和密钥
- PostgreSQL
- Redis
- Qdrant
- Embedding
- TTS/RVC 路径

`config.py` 不提供业务配置回退值。所有环境变量键必须存在；密钥可以为空，但不能缺少键。

### 7.3 Python 和前端依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt

cd ..\frontend
npm ci
```

`pip install` 只安装 Python 包，不下载模型。

### 7.4 下载公开模型

```powershell
cd ..\backend
.\.venv\Scripts\Activate.ps1
python scripts\download_models.py
```

单独下载：

```powershell
python scripts\download_models.py --only cosyvoice --source modelscope
python scripts\download_models.py --only rvc
```

公开资源包括 CosyVoice2、HuBERT 和 RMVPE。用户自己的 RVC 模型不下载：

```text
models/rvc/weights/*.pth
models/rvc/indexes/*.index
```

### 7.5 初始化基础设施

```powershell
docker compose --env-file .env up -d postgres redis qdrant
.\.venv\Scripts\python.exe scripts\migrate.py
.\.venv\Scripts\python.exe scripts\init_qdrant.py
```

### 7.6 启动开发环境

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

另一个终端：

```powershell
cd ..\frontend
npm run dev
```

也可以运行：

```powershell
.\scripts\start-dev.ps1
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`

## 8. Docker

前后端位于同级目录，并准备好 `.env` 和模型后：

```powershell
.\scripts\docker-up.ps1
```

默认访问：`http://127.0.0.1:8080`

```powershell
docker compose --env-file .env ps
docker compose --env-file .env logs -f backend frontend
docker compose --env-file .env down
```

`down` 不删除命名卷；`down -v` 会删除数据库和向量数据。

## 9. 数据和缓存

### PostgreSQL

| 表 | 用途 |
|---|---|
| `sessions` | 会话和身份 |
| `messages` | 用户及助手消息 |
| `documents` | 文档元数据 |
| `document_chunks` | 文档切片 |
| `user_facts` | 长期记忆 |
| `app_settings` | 全局提示配置 |
| `client_branding` | 品牌配置 |
| `app_assets` | 旧资源兼容 |
| `schema_migrations` | 迁移版本和校验和 |

### Redis

| Key | 用途 |
|---|---|
| `session:<id>:messages` | 最近消息 |
| `session:<id>:meta` | 标题和更新时间 |
| `message:<id>:content` | TTS 正文 |

Redis 不可用时回源 PostgreSQL；按消息 ID 查询成功后回填 Redis。

### Qdrant

- 知识库集合：`QDRANT_COLLECTION`
- 记忆集合：`MEMORY_COLLECTION`
- 维度：`EMBEDDING_DIM`

## 10. API 概览

完整定义见 `/docs`。

### 系统与设置

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET/PUT/DELETE | `/api/settings/prompt` | 人格提示 |
| GET/PUT | `/api/settings/branding` | 品牌设置 |
| PUT | `/api/settings/branding/background` | 背景设置 |
| POST/GET/DELETE | `/api/settings/branding/logo` | Logo |
| POST/GET | `/api/settings/branding/background/video` | 背景视频 |
| GET | `/api/settings/branding/media` | 媒体库 |
| POST | `/api/settings/branding/media/use` | 使用媒体 |

### 聊天与会话

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 普通聊天 |
| POST | `/api/chat/stream` | SSE 流式聊天 |
| GET | `/api/sessions` | 会话列表 |
| GET/DELETE | `/api/sessions/{id}` | 会话详情/删除 |

### 记忆与知识库

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/memories` | 记忆列表 |
| DELETE | `/api/memories/{id}` | 删除记忆 |
| POST | `/api/documents/upload` | 上传 Markdown/TXT/PDF |
| GET | `/api/documents` | 文档列表 |
| DELETE | `/api/documents/{id}` | 删除文档 |
| GET | `/api/rag/search?q=` | 检索 |

### 音频与 RVC

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/audio/engines` | 引擎状态 |
| GET | `/api/audio/voices` | 音色列表 |
| POST | `/api/audio/tts/stream` | 按消息 ID 流式合成 |
| POST | `/api/audio/rvc/runtime` | 启用/关闭 RVC |
| GET | `/api/audio/rvc/models` | 模型列表 |
| POST | `/api/audio/rvc/preload` | 预加载 |
| POST | `/api/audio/rvc/models` | 上传模型/索引 |
| DELETE | `/api/audio/rvc/models/{name}` | 删除模型 |
| POST | `/api/audio/rvc/refresh` | 刷新模型 |

## 11. 音频协议

音频事件：

```json
{
  "type": "audio",
  "index": 0,
  "text": "当前段文本",
  "engine": "cosyvoice",
  "format": "pcm_s16le",
  "sample_rate": 24000,
  "channels": 1,
  "data": "<base64 PCM>"
}
```

结束和错误：

```json
{"type":"done","engine":"cosyvoice"}
{"type":"error","code":"tts_stream_failed","message":"..."}
```

前端把 PCM 转换为 `Float32Array`，按段调度到 Web Audio 时间轴。

## 12. 验证

后端：

```powershell
.\.venv\Scripts\python.exe -B -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in Path('app').rglob('*.py')]"
```

前端：

```powershell
cd ..\frontend
npx vue-tsc --noEmit -p tsconfig.json
npm test
```

冒烟测试：

```powershell
cd ..\backend
.\scripts\verify.ps1
```

## 13. 常见问题

### 没有语音

检查 `TTS_ENABLED`、`/api/audio/engines`、`/api/audio/voices` 和后端模型加载日志。

### 消息不存在

前后端必须同步更新。当前 TTS 只接受数据库中的助手 UUID，旧前端发送 `text` 会收到 `message_id_required`。

### 长回答后半段没读

旧版本限制 800 字符。当前上限为 20,000，更新并重启后端。

### 模型下载失败

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:<port>"
$env:HTTPS_PROXY = "http://127.0.0.1:<port>"
python scripts\download_models.py
```

国内网络可尝试：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts\download_models.py --only rvc
```

### Git 无法连接 GitHub 443

代理客户端的“系统代理”不一定作用于 Git Bash：

```bash
git config --global http.proxy http://127.0.0.1:<port>
git config --global https.proxy http://127.0.0.1:<port>
git ls-remote origin
```

### GPU 显存不足

- 关闭 RVC 先验证 CosyVoice。
- 关闭旧后端进程。
- 避免并发语音请求。
- 检查模型加载日志。

### RVC 首次使用慢

确认预加载接口成功：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/audio/rvc/preload?model_name=<name>"
```

### 声音有电音或段间接缝异常

区分来源：

- 纯 CosyVoice 也有：检查源模型和参考音频。
- 仅 RVC 后出现：检查 RVC 模型、索引、F0、精度和 GPU。
- 仅段间出现：检查分段大小、预加载和播放调度。
- 每个字都有金属感：通常是 RVC 模型或声码器质量。

当前 CUDA 模型可能使用 FP16。若怀疑半精度失真，可将 `audio.py` 中 CosyVoice 的 `fp16` 和 RVC 的 `is_half` 改为 `False`，代价是更高显存和更慢加载。

## 14. 安全与运维

- `.env` 含密钥，不进入 Git。
- 当前没有登录和权限系统，禁止直接公网暴露。
- `.pth`、`.index` 和模型文件不进入 Git。
- 日志不得记录 API Key、密码、验证码或完整敏感内容。
- PostgreSQL、Redis、Qdrant 本地端口默认只绑定 `127.0.0.1`。
- 生产环境必须配置 HTTPS、访问控制和备份。

## 15. 开发约定

- 新环境变量同步更新 `.env.example`。
- `config.py` 不添加静默业务回退值。
- 数据库变更使用迁移，不修改已执行迁移。
- TTS 不自动切换引擎。
- 修改音频协议时同步更新前后端。
- 修改消息 ID 协议时，流式完成事件必须返回真实数据库 ID。

## 16. 后续改造

1. 正式登录替换全局身份。
2. 增加 RVC 模型说话人和质量元数据。
3. 增加 readiness 接口报告模型加载状态。
4. 增加音频端到端测试。
5. 增加长文本 TTS 取消、恢复和缓存。
6. 增加模型 SHA-256 校验。
7. 按用户隔离 Redis 消息缓存 key。
