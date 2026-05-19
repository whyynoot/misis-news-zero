param(
    [int]$Limit = 50,
    [ValidateSet("bert", "llm")]
    [string]$Engine = "llm",
    [switch]$Force,
    [switch]$Summarize,
    [switch]$Local
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$manageArgs = @("manage.py", "run_monitoring", "--no-bootstrap", "--limit", "$Limit", "--engine", $Engine)
if ($Force) { $manageArgs += "--force" }
if ($Summarize -or $Engine -eq "llm") { $manageArgs += "--summarize" }

if ($Local) {
    & python @manageArgs
    exit $LASTEXITCODE
}

& docker compose exec -T web python @manageArgs
exit $LASTEXITCODE
