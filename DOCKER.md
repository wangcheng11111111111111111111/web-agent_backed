# Nebula Docker 部署

本文档描述如何在一台带 NVIDIA GPU 的机器上用 Docker Compose 启动完整应用，包括：

- Vue 3 前端
- FastAPI 后端
- PostgreSQL
- Redis
- Qdrant
- 数据库迁移与 Qdrant 集合初始化
- Edge TTS / CosyVoice2 / RVC 语音链路

## 0. 两个仓库的检出方式

前后端分别维护为两个独立仓库。运行 Docker Compose 时，请在同一父目录中检出：

```text
workspace/
├── backend/   web-agent_backed
└── frontend/  web-agent_fronted
```

```powershell
git clone https://github.com/wangcheng11111111111111111111/web-agent_backed.git backend
git clone https://github.com/wangcheng11111111111111111111/web-agent_fronted.git frontend
cd backend
```

后端仓库内的 `compose.yaml` 默认通过 `../frontend` 构建前端。如果目录名不同，请在
`.env` 中修改 `FRONTEND_CONTEXT`。以下命令均在后端仓库根目录执行。

## 1. 架构

```text
Browser
  │  http://127.0.0.1:8080
  ▼
frontend (Nginx)
  │  /api, /docs, /openapi.json
  ▼
backend (FastAPI + CUDA)
  ├── postgres
  ├── redis
  └── qdrant

init
  └── PostgreSQL migrations + Qdrant collections
```

宿主机只需要发布前端端口。后端通过 Nginx 同源代理访问：

```text
http://127.0.0.1:8080
http://127.0.0.1:8080/docs
http://127.0.0.1:8080/api/health
```

## 2. 环境要求

- Windows 10/11 或 Linux
- Docker Desktop / Docker Engine
- Docker Compose v2
- NVIDIA 显卡和驱动
- NVIDIA Container Toolkit，或 Docker Desktop 的 WSL2 GPU 支持
- 至少 10 GB 可用显存
- 建议至少 32 GB 内存和 20 GB 空闲磁盘

先确认 Docker 支持 GPU：

```powershell
docker info
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

如果第二条命令无法看到 GPU，后端无法加载 CosyVoice2 和 RVC。

## 3. 模型目录

大模型不进入 Docker 镜像，而是从宿主机挂载到容器。

### 3.1 CosyVoice2

```text
models/cosyvoice2/
├── cosyvoice2.yaml
├── llm.pt
├── flow.pt
├── hift.pt
├── campplus.onnx
├── speech_tokenizer_v2.onnx
└── CosyVoice-BlankEN/
```

### 3.2 RVC 通用推理资源

```text
vendor/RVC/assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
└── rmvpe/
    └── rmvpe.pt
```

这些资源来自 `lj1995/VoiceConversionWebUI`，不是用户训练出来的音色。

安装 `requirements-cosyvoice.txt` 后，可以使用仓库内脚本下载模型：

```powershell
python scripts/download_models.py
```

脚本优先从 ModelScope 下载 CosyVoice2，并可从 Hugging Face 下载 HuBERT 和 RMVPE。
已有文件会复用，RVC 用户训练的 `.pth`、`.index` 仍需自行放入下一步目录。

### 3.3 RVC 自定义音色

```text
models/rvc/
├── weights/
│   └── feiouni.pth
└── indexes/
    └── feiouni_added_IVF549_Flat_nprobe_1_feiouni_v2.index
```

`.pth` 和 `.index` 可以在前端自由组合。索引也可以留空，由后端自动选择同名最新文件。

## 4. 环境变量

复制模板：

```powershell
Copy-Item .env.example .env
```

至少配置：

```ini
LLM_API_KEY=你的聊天模型密钥
EMBEDDING_API_KEY=你的向量模型密钥
PG_PASSWORD=数据库密码
```

`scripts/docker-up.ps1` 和 `docker compose` 使用：

```text
--env-file .env
```

因此在 `.env` 中设置一次 `PG_PASSWORD`，PostgreSQL 和 FastAPI 会使用相同密码。

语气参考音频默认使用本地 Whisper Small 自动识别，`STT_DEVICE=auto` 会优先使用
CUDA。若配置远程 STT，可在
`.env` 增加：

```ini
STT_ENABLED=true
STT_DEVICE=auto
STT_WHISPER_MODEL=small
STT_BASE_URL=https://api.siliconflow.cn/v1
STT_API_KEY=你的语音识别密钥
STT_MODEL=FunAudioLLM/SenseVoiceSmall
STT_TIMEOUT=30
```

远程识别失败不会丢失上传音频，系统会自动尝试本地 Whisper；仍失败时预设会标记为
“待补全文本”，可在前端手填。参考音频建议 3–15 秒，最长 30 秒。

Compose 会自动覆盖容器内网络地址：

```text
PG_HOST=postgres
REDIS_HOST=redis
QDRANT_HOST=qdrant
```

不需要手工修改为 Docker 主机名。

## 5. 一键启动

Windows：

```powershell
.\scripts\docker-up.ps1
```

Linux / macOS：

```bash
sh scripts/docker-up.sh
```

脚本会：

1. 检查 Docker
2. 检查 `.env`
3. 检查 CosyVoice2、HuBERT、RMVPE 必需文件
4. 构建前后端镜像
5. 启动数据库依赖
6. 执行迁移和 Qdrant 初始化
7. 启动 GPU 后端和 Nginx 前端

也可以直接运行：

```powershell
docker compose --env-file .env up --build -d
```

首次构建需要下载 PyTorch、CUDA 依赖和构建 Python 包，耗时会比较长。

## 6. 服务说明

| 服务 | 镜像 | 说明 |
|---|---|---|
| `postgres` | `postgres:16-alpine` | 会话、消息、文档、长期事实 |
| `redis` | `redis:7-alpine` | 最近消息缓存 |
| `qdrant` | `qdrant/qdrant:v1.12.4` | 知识库与长期记忆向量 |
| `init` | 本地后端镜像 | 一次性迁移和集合初始化 |
| `backend` | `nebula-backend:local` | FastAPI、CosyVoice2、Edge TTS、RVC |
| `frontend` | `nebula-frontend:local` | Vue 静态资源与 API 反向代理 |

默认只发布前端端口：

```text
8080 -> frontend:80
```

PostgreSQL、Redis、Qdrant 和 FastAPI 仅位于 Compose 内部网络。

修改端口：

```powershell
$env:WEB_PORT = "8090"
.\scripts\docker-up.ps1
```

## 7. 持久化数据

Compose 命名卷：

```text
postgres_data
redis_data
qdrant_data
```

宿主机目录：

```text
models/                模型、RVC 权重与索引
vendor/RVC/assets/     HuBERT 与 RMVPE
data/                  品牌媒体、语气参考音频、Hugging Face 缓存
```

其中 `data/voice_prompts/prompts.json` 保存语气预设元数据，同目录的其他
音频文件是上传的参考片段。Compose 已挂载整个 `data/`，重建容器不会丢失。

删除容器不会删除模型和数据库卷：

```powershell
docker compose --env-file .env down
```

清除全部数据库和 Qdrant 数据：

```powershell
docker compose --env-file .env down -v
```

`-v` 会删除数据库和向量数据，执行前应确认已有备份。

## 8. 常用运维

查看状态：

```powershell
docker compose --env-file .env ps
```

查看日志：

```powershell
docker compose --env-file .env logs -f backend frontend
```

只重启后端：

```powershell
docker compose --env-file .env restart backend
```

代码更新后重建：

```powershell
docker compose --env-file .env up --build -d
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health
```

## 9. 依赖结构

- `requirements.txt`：Web API、数据库、RAG、Edge TTS。
- `requirements-cosyvoice.txt`：CosyVoice2、Matcha-TTS、RVC 共用依赖。
- `requirements-docker.txt`：CUDA 版 PyTorch、音频处理与 RVC 依赖。
- `frontend/package-lock.json`：前端可复现安装。

镜像构建使用 `npm ci`，不会安装开发机上的 `node_modules`。

## 10. 故障排查

### GPU 不可见

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

确认 Docker Desktop 使用 WSL2 后端，并安装支持 CUDA 的最新 NVIDIA 驱动。

### 缺少模型

重新运行一键脚本，它会列出缺失文件。模型目录结构见本文第 3 节。

### `init` 服务失败

```powershell
docker compose --env-file .env logs init
```

通常原因是 `PG_PASSWORD` 不一致、PostgreSQL 未就绪，或 `EMBEDDING_API_KEY` 未配置。

### 后端健康但语音不可用

检查：

```powershell
docker compose --env-file .env logs backend
```

CosyVoice2 首次预热会占用较长时间，但 `/api/health` 仍应保持 `200`。

### 8080 端口被占用

设置新的 `WEB_PORT` 后重新运行一键脚本。
