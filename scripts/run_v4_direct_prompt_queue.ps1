param(
  [int]$Limit = 237,
  [int]$NumCtx = 16384,
  [int]$MaxTokens = 4096,
  [int]$TextLimit = 2200,
  [switch]$PullMissing
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Dataset = Join-Path $Root "analysis_outputs\FINAL_v4_clean_social_signal_dataset_965_normalized.csv"
$LogDir = Join-Path $Root "analysis_outputs\v4_direct_prompt_queue_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

python scripts\prepare_v4_clean_dataset.py
python scripts\build_v4_direct_prompt_search_assets.py

$Models = @(
  @{ Name = "qwen3.5:9b"; Tag = "qwen35_9b"; Pull = $false },
  @{ Name = "gemma4:e4b"; Tag = "gemma4_e4b"; Pull = $false },
  @{ Name = "batiai/qwen3.6-35b:iq3"; Tag = "qwen36_35b_iq3"; Pull = $false },
  @{ Name = "qwen3:14b"; Tag = "qwen3_14b"; Pull = $true },
  @{ Name = "qwen3.5:27b"; Tag = "qwen35_27b"; Pull = $true },
  @{ Name = "qwen3.6:27b"; Tag = "qwen36_27b"; Pull = $true }
)

$Prompts = @(
  "direct_strict",
  "direct_recall",
  "direct_checklist",
  "direct_negative",
  "direct_fewshot",
  "direct_minimal",
  "direct_taxonomy",
  "direct_precision",
  "broad_balanced",
  "broad_recall",
  "broad_weak_context"
)

$Installed = (ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split "\s+")[0] })

foreach ($Model in $Models) {
  if ($Installed -notcontains $Model.Name) {
    if ($PullMissing -and $Model.Pull) {
      $pullLog = Join-Path $LogDir ("pull_{0}.log" -f $Model.Tag)
      "pulling $($Model.Name)" | Tee-Object -FilePath $pullLog -Append
      ollama pull $Model.Name *>> $pullLog
      $Installed = (ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split "\s+")[0] })
    }
    if ($Installed -notcontains $Model.Name) {
      "SKIP missing model $($Model.Name)" | Tee-Object -FilePath (Join-Path $LogDir "missing_models.log") -Append
      continue
    }
  }

  foreach ($Prompt in $Prompts) {
    $PromptFile = Join-Path $Root ("prompts\v4_direct_search\{0}.md" -f $Prompt)
    $Tag = ("{0}_{1}_think_ctx{2}_tok{3}_n{4}" -f $Model.Tag, $Prompt, $NumCtx, $MaxTokens, $Limit)
    $Stdout = Join-Path $LogDir ("{0}.stdout.log" -f $Tag)
    $Stderr = Join-Path $LogDir ("{0}.stderr.log" -f $Tag)
    "RUN $($Model.Name) $Prompt limit=$Limit" | Tee-Object -FilePath $Stdout -Append
    python scripts\run_v4_social_signal_prompt_experiment.py `
      --dataset $Dataset `
      --prompt-file $PromptFile `
      --limit $Limit `
      --ids-from $Dataset `
      --batch-size 1 `
      --model $Model.Name `
      --think `
      --num-ctx $NumCtx `
      --max-tokens $MaxTokens `
      --text-limit $TextLimit `
      --timeout 900 `
      --tag $Tag `
      *>> $Stdout
    if ($LASTEXITCODE -ne 0) {
      "FAILED $($Model.Name) $Prompt exit=$LASTEXITCODE" | Tee-Object -FilePath $Stderr -Append
    }
    python scripts\evaluate_v4_strength_metrics.py *>> (Join-Path $LogDir "evaluate_after_each_run.log")
  }
}

python scripts\evaluate_v4_strength_metrics.py
