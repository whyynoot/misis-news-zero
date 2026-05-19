param(
    [int]$Tail = 20
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$logPath = Join-Path $ProjectRoot "tmp-ui-checks\history-backfill.out.log"
$errPath = Join-Path $ProjectRoot "tmp-ui-checks\history-backfill.err.log"

Write-Output "Historical backfill log:"
if (Test-Path $logPath) {
    Get-Content -Encoding UTF8 -Tail $Tail $logPath
} else {
    Write-Output "No history-backfill.out.log yet."
}

Write-Output ""
Write-Output "Recent warnings/errors:"
if (Test-Path $errPath) {
    Get-Content -Encoding UTF8 -Tail $Tail $errPath
} else {
    Write-Output "No history-backfill.err.log yet."
}

Write-Output ""
Write-Output "Database counters:"
docker compose exec -T web python manage.py shell -c "from analyzer.models import NewsItem, NewsClassification, DailySentimentSummary; print('real_news', NewsItem.objects.filter(is_seed=False).count()); print('llm_classifications', NewsClassification.objects.filter(engine='llm').count()); print('llm_days', DailySentimentSummary.objects.filter(engine='llm').count())"
