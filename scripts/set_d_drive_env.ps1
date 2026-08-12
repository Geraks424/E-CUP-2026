# Force all caches, temp, models, and pip storage onto D: (E-CUP 2026 workspace).
# Dot-source before GPU baseline work: . .\scripts\set_d_drive_env.ps1

$Root = "D:\E-CUP 2026"

$env:HF_HOME                    = "$Root\.cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE     = "$Root\.cache\huggingface\hub"
$env:TRANSFORMERS_CACHE         = "$Root\.cache\huggingface\transformers"
$env:TORCH_HOME                 = "$Root\.cache\torch"
$env:TMP                        = "$Root\.tmp"
$env:TEMP                       = "$Root\.tmp"
$env:PIP_CACHE_DIR              = "$Root\.cache\pip"
$env:SHARED_MODELS_PATH          = "$Root\shared_models"

foreach ($dir in @(
    $env:HF_HOME,
    $env:HUGGINGFACE_HUB_CACHE,
    $env:TRANSFORMERS_CACHE,
    $env:TORCH_HOME,
    $env:TMP,
    $env:PIP_CACHE_DIR,
    $env:SHARED_MODELS_PATH,
    "$Root\local_data"
)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "D:-only env active:"
Write-Host "  HF_HOME=$env:HF_HOME"
Write-Host "  SHARED_MODELS_PATH=$env:SHARED_MODELS_PATH"
Write-Host "  TMP/TEMP=$env:TMP"
Write-Host "  PIP_CACHE_DIR=$env:PIP_CACHE_DIR"
