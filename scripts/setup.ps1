$ErrorActionPreference = "Stop"
$Backend = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Backend
$Frontend = Join-Path $ProjectRoot "frontend"

Write-Host "1/5 准备后端环境变量"
$EnvFile = Join-Path $Backend ".env"
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath (Join-Path $Backend ".env.example") -Destination $EnvFile
    Write-Host "已创建 .env，请填写 LLM_API_KEY、PG_PASSWORD 和 EMBEDDING_API_KEY"
}

Write-Host "2/5 启动 Redis、PostgreSQL 与 Qdrant"
Push-Location $Backend
try {
    docker compose `
        --env-file (Join-Path $Backend ".env") `
        -f (Join-Path $Backend "compose.yaml") `
        up -d postgres redis qdrant
}
finally {
    Pop-Location
}

Write-Host "3/5 安装后端依赖"
Push-Location $Backend
try {
    if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        python -m venv .venv
    }
    & ".venv\Scripts\python.exe" -m pip install -r requirements.txt
}
finally {
    Pop-Location
}

Write-Host "4/5 初始化数据库与 Qdrant 集合"
Push-Location $Backend
try {
    & ".venv\Scripts\python.exe" "scripts\migrate.py"
    & ".venv\Scripts\python.exe" "scripts\init_qdrant.py"
}
finally {
    Pop-Location
}

Write-Host "5/5 安装前端依赖"
Push-Location $Frontend
try {
    npm ci
}
finally {
    Pop-Location
}

Write-Host "初始化完成。请分别在两个终端启动后端和前端。"
