param(
    [double]$IntervalHours = 4,
    [int]$LiveLimit = 50,
    [int]$HistoryDays = 365,
    [int]$HistoryLimitPerDay = 8,
    [string]$HistorySources = "interfax",
    [switch]$HistoryOnStart,
    [switch]$Local
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $ProjectRoot "tmp-ui-checks"
$LogPath = Join-Path $LogDir "llm-worker.log"
$StopFile = Join-Path $LogDir "llm-worker.stop"
$LockFile = Join-Path $LogDir "llm-worker.lock"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Remove-Item -LiteralPath $StopFile -ErrorAction SilentlyContinue

if (Test-Path $LockFile) {
    $existingPid = (Get-Content $LockFile -Raw -ErrorAction SilentlyContinue).Trim()
    if ($existingPid -match "^[0-9]+$" -and (Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue)) {
        "LLM monitoring worker already running (pid=$existingPid)." | Tee-Object -Append $LogPath
        exit 0
    }
}

Set-Content -Path $LockFile -Value "$PID" -Encoding UTF8

$env:PYTHONIOENCODING = "utf-8"
$env:LLM_ENABLED = "True"

function Set-EnvDefault {
    param([string]$Name, [string]$Value)
    if (-not [Environment]::GetEnvironmentVariable($Name)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

Set-EnvDefault "LLM_BASE_URL" "http://localhost:11434"
Set-EnvDefault "LLM_MODEL" "gemma4:e2b"
Set-EnvDefault "LLM_TIMEOUT_SECONDS" "180"
Set-EnvDefault "LLM_TEMPERATURE" "0"
Set-EnvDefault "LLM_MAX_TOKENS" "2048"
Set-EnvDefault "LLM_BATCH_NEWS_SIZE" "1"
Set-EnvDefault "LLM_CONCURRENCY" "1"
Set-EnvDefault "LLM_THINK" "False"

function Write-WorkerLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Encoding UTF8 -Value "[$timestamp] $Message"
}

function Invoke-LoggedCommand {
    param([string[]]$CommandArgs)
    Write-WorkerLog ("RUN python " + ($CommandArgs -join " "))
    $output = & python @CommandArgs 2>&1
    if ($output) {
        $output | ForEach-Object { Add-Content -Path $LogPath -Encoding UTF8 -Value "$_" }
    }
    $exitCode = $LASTEXITCODE
    Write-WorkerLog ("EXIT code " + $exitCode)
    return $exitCode
}

try {
    Write-WorkerLog "LLM worker started. model=$env:LLM_MODEL, interval=${IntervalHours}h, live=$LiveLimit."

    if ($HistoryOnStart) {
        Invoke-LoggedCommand @(
            "manage.py", "backfill_news_history",
            "--days", "$HistoryDays",
            "--sources", $HistorySources,
            "--limit-per-day", "$HistoryLimitPerDay",
            "--engine", "llm",
            "--summarize",
            "--period", "day"
        )
    }

    while (-not (Test-Path $StopFile)) {
        Invoke-LoggedCommand @(
            "manage.py", "run_monitoring",
            "--engine", "llm",
            "--summarize",
            "--period", "day",
            "--limit", "$LiveLimit",
            "--no-bootstrap"
        )

        $sleepSeconds = [Math]::Max([int]($IntervalHours * 3600), 60)
        Write-WorkerLog "Sleeping for $sleepSeconds seconds."
        for ($elapsed = 0; $elapsed -lt $sleepSeconds; $elapsed += 30) {
            if (Test-Path $StopFile) { break }
            Start-Sleep -Seconds 30
        }
    }
}
finally {
    Remove-Item -LiteralPath $LockFile -ErrorAction SilentlyContinue
}

Write-WorkerLog "LLM worker stopped."
