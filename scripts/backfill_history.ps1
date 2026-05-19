param(
    [int]$Days = 365,
    [string]$FromDate = "",
    [string]$ToDate = "",
    [string]$Sources = "interfax",
    [int]$LimitPerDay = 8,
    [int]$ChunkDays = 1,
    [int]$Limit = 0,
    [ValidateSet("bert", "llm")]
    [string]$Engine = "llm",
    [double]$DelaySeconds = 0.25,
    [switch]$Force,
    [switch]$Summarize,
    [switch]$CleanSeed,
    [switch]$Local
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$manageArgs = @(
    "manage.py",
    "backfill_news_history",
    "--sources", $Sources,
    "--limit-per-day", "$LimitPerDay",
    "--chunk-days", "$ChunkDays",
    "--engine", $Engine,
    "--delay-seconds", "$DelaySeconds"
)

if ($FromDate) {
    $manageArgs += @("--from-date", $FromDate)
} else {
    $manageArgs += @("--days", "$Days")
}

if ($ToDate) { $manageArgs += @("--to-date", $ToDate) }
if ($Limit -gt 0) { $manageArgs += @("--limit", "$Limit") }
if ($Force) { $manageArgs += "--force" }
if ($Summarize) { $manageArgs += "--summarize" }
if ($CleanSeed) { $manageArgs += "--clean-seed" }

if ($Local) {
    & python @manageArgs
    exit $LASTEXITCODE
}

& docker compose exec -T web python @manageArgs
exit $LASTEXITCODE
