param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if ($Build) {
    docker compose up -d --build db web scheduler
} else {
    docker compose up -d db web scheduler
}

docker compose ps
