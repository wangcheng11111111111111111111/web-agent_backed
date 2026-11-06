$ErrorActionPreference = "Stop"
$Backend = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Backend
$Frontend = Join-Path $ProjectRoot "frontend"
$LogDir = Join-Path $Backend "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$BackendProcess = Start-Process `
    -FilePath (Join-Path $Backend ".venv\Scripts\python.exe") `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload") `
    -WorkingDirectory $Backend `
    -RedirectStandardOutput (Join-Path $LogDir "backend.out.log") `
    -RedirectStandardError (Join-Path $LogDir "backend.err.log") `
    -WindowStyle Hidden `
    -PassThru

$FrontendProcess = Start-Process `
    -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") `
    -WorkingDirectory $Frontend `
    -RedirectStandardOutput (Join-Path $LogDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $LogDir "frontend.err.log") `
    -WindowStyle Hidden `
    -PassThru

Write-Host "后端 PID: $($BackendProcess.Id)"
Write-Host "前端 PID: $($FrontendProcess.Id)"
Write-Host "前端: http://127.0.0.1:5173"
Write-Host "后端文档: http://127.0.0.1:8000/docs"
Write-Host "日志目录: $LogDir"
