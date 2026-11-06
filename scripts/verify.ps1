$ErrorActionPreference = "Stop"
$BaseUrl = "http://127.0.0.1:8000"
$ClientId = "verify-client"

$Health = Invoke-RestMethod -Uri "$BaseUrl/api/health"
if ($Health.status -ne "ok") {
    throw "健康检查失败"
}
Write-Host "健康检查: OK"

$Docs = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/docs"
if ($Docs.StatusCode -ne 200) {
    throw "Swagger UI 不可用"
}
Write-Host "Swagger UI: OK"

$Payload = @{ message = "现在几点？" } | ConvertTo-Json
try {
    $Stream = Invoke-WebRequest `
        -UseBasicParsing `
        -Method Post `
        -Uri "$BaseUrl/api/chat/stream" `
        -ContentType "application/json" `
        -Headers @{ Accept = "text/event-stream"; "X-Client-ID" = $ClientId } `
        -Body $Payload `
        -TimeoutSec 90
    Write-Host "SSE HTTP $($Stream.StatusCode)"
    Write-Host $Stream.Content
}
catch {
    $Response = $_.Exception.Response
    if ($Response) {
        Write-Host "SSE HTTP $([int]$Response.StatusCode)"
        Write-Host $Response.Content.ReadAsStringAsync().Result
    }
    throw
}

if ($Stream.Content -notmatch '"type":\s*"done"') {
    throw "SSE 没有收到结束事件"
}
if ($Stream.Content -notmatch '"stage":\s*"load_memory"') {
    throw "SSE 没有执行长期记忆加载节点"
}
Write-Host "SSE 结束事件: OK"
Write-Host "长期记忆加载节点: OK"
