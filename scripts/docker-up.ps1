$ErrorActionPreference = "Stop"

$Backend = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Backend
$Frontend = Join-Path $ProjectRoot "frontend"
$EnvFile = Join-Path $Backend ".env"
$EnvExample = Join-Path $Backend ".env.example"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "未找到 Docker。请安装并启动 Docker Desktop，并启用 WSL2 与 NVIDIA GPU 支持。"
}

if (-not (Test-Path -LiteralPath (Join-Path $Frontend "Dockerfile"))) {
    throw "未找到同级前端仓库：$Frontend。请将 web-agent_fronted 克隆为 backend 的同级 frontend 目录。"
}

& docker compose version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "未找到 Docker Compose v2。"
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    throw "已创建 .env。请填写 LLM_API_KEY、PG_PASSWORD、EMBEDDING_API_KEY 后重新运行。"
}

$requiredFiles = @(
    "models/cosyvoice2/cosyvoice2.yaml",
    "models/cosyvoice2/llm.pt",
    "models/cosyvoice2/flow.pt",
    "models/cosyvoice2/hift.pt",
    "models/cosyvoice2/campplus.onnx",
    "models/cosyvoice2/speech_tokenizer_v2.onnx",
    "models/cosyvoice2/CosyVoice-BlankEN/config.json",
    "vendor/RVC/assets/hubert_base/config.json",
    "vendor/RVC/assets/hubert_base/pytorch_model.bin",
    "vendor/RVC/assets/rmvpe/rmvpe.pt"
)
$missing = @(
    $requiredFiles | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $Backend $_))
    }
)
if ($missing.Count -gt 0) {
    throw "缺少语音模型文件：" + ($missing -join ", ") + "。请按照 README.md 准备模型目录。"
}

$rvcModels = @(Get-ChildItem -LiteralPath (Join-Path $Backend "models/rvc/weights") -Filter "*.pth" -File -ErrorAction SilentlyContinue)
if ($rvcModels.Count -eq 0) {
    Write-Warning "未发现 RVC .pth 模型。容器仍可启动，但 RVC 本地音色不可用。"
}

Push-Location $Backend
try {
    & docker compose --env-file $EnvFile up --build -d
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose 启动失败。"
    }
    & docker compose --env-file $EnvFile ps
} finally {
    Pop-Location
}

$webPort = if ($env:WEB_PORT) { $env:WEB_PORT } else { "8080" }
$webPortSetting = Get-Content -LiteralPath $EnvFile |
    Where-Object { $_ -match "^\s*WEB_PORT\s*=" } |
    Select-Object -Last 1
if (-not $env:WEB_PORT -and $webPortSetting) {
    $webPort = ($webPortSetting -split "=", 2)[1].Trim()
}
Write-Host ""
Write-Host "Nebula 已启动：http://127.0.0.1:$webPort"
Write-Host "查看日志：docker compose --env-file .env logs -f backend frontend"
