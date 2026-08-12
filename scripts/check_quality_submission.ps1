# Phase 0 local check for Task 2 quality baseline (dry-run by default).
param(
    [string]$DataCsv = "D:\data.csv",
    [string]$ImagesDir = "",
    [int]$SubsetSize = 200,
    [int]$Seed = 42,
    [ValidateSet("dry-run", "full")]
    [string]$Mode = "dry-run"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LocalData = Join-Path $RepoRoot "local_data"
$ReportsDir = Join-Path $RepoRoot "reports\baseline"
$BaselineDir = Join-Path $RepoRoot "baseline\quality-baseline-submit"
$SubsetCsv = Join-Path $LocalData "quality_subset.csv"
$SmokeOut = Join-Path $LocalData "smoke_submit.csv"
$ScoreReport = Join-Path $ReportsDir "phase0-baseline.json"
$RuntimeReport = Join-Path $ReportsDir "phase0-runtime.json"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Test-PythonModule([string]$ModuleName) {
    python -c "import $ModuleName" 2>$null
    return $LASTEXITCODE -eq 0
}

Write-Step "Preflight"
if (-not (Test-Path $DataCsv)) {
    throw "Data CSV not found: $DataCsv"
}
if (-not (Test-Path $BaselineDir)) {
    throw "Baseline dir not found: $BaselineDir"
}

$requiredModules = @("pandas", "sklearn", "joblib")
foreach ($mod in $requiredModules) {
    if (-not (Test-PythonModule $mod)) {
        throw "Missing Python module '$mod'. Install: pip install -r requirements-dev.txt"
    }
}

$metadataPath = Join-Path $BaselineDir "metadata.json"
if (-not (Test-Path $metadataPath)) {
    throw "metadata.json missing in baseline submit root"
}
$metadata = Get-Content $metadataPath -Raw | ConvertFrom-Json
if (-not $metadata.entry_point) {
    throw "metadata.json missing entry_point"
}

$classifierPath = Join-Path $BaselineDir "baseline_qwen3vl_bf16.joblib"
if (-not (Test-Path $classifierPath)) {
    throw "Classifier joblib missing: $classifierPath"
}

$forbiddenPatterns = @("*.safetensors", "*.bin", "*.pt", "*.pth", "data.csv", "images.zip")
foreach ($pattern in $forbiddenPatterns) {
    $hits = Get-ChildItem -Path $RepoRoot -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\\.git\\" }
    if ($hits) {
        Write-Warning "Found potentially forbidden large artifact: $($hits[0].FullName)"
    }
}

Write-Step "Prepare subset"
$prepareArgs = @(
    "scripts/prepare_quality_subset.py",
    "--data_csv", $DataCsv,
    "--size", $SubsetSize,
    "--seed", $Seed,
    "--output", $SubsetCsv
)
if ($ImagesDir) {
    $prepareArgs += @("--images_dir", $ImagesDir)
}
python @prepareArgs
if ($LASTEXITCODE -ne 0) { throw "prepare_quality_subset failed" }

if ($Mode -eq "full") {
    Write-Host "`nNOTE: full baseline inference requires CUDA + SHARED_MODELS_PATH models." -ForegroundColor Yellow
    Write-Host "Run inside Docker image $($metadata.image) with GPU." -ForegroundColor Yellow
    exit 0
}

Write-Step "Dry-run smoke submission"
$smokeArgs = @(
    "scripts/smoke_quality_baseline.py",
    "--input_csv", $SubsetCsv,
    "--output_csv", $SmokeOut,
    "--baseline_dir", $BaselineDir
)
if ($ImagesDir) {
    $smokeArgs += @("--images_dir", $ImagesDir)
}
python @smokeArgs
if ($LASTEXITCODE -ne 0) { throw "smoke_quality_baseline failed" }

Write-Step "Validate submission format"
python scripts/validate_quality_submission.py --input_csv $SubsetCsv --submission_csv $SmokeOut
if ($LASTEXITCODE -ne 0) { throw "validate_quality_submission failed" }

Write-Step "Offline score (dry-run / label oracle)"
python scripts/score_quality_offline.py `
    --input_csv $SubsetCsv `
    --submission_csv $SmokeOut `
    --report $ScoreReport `
    --mode dry_run
if ($LASTEXITCODE -ne 0) { throw "score_quality_offline failed" }

Write-Step "Write runtime limits report"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
$runtime = @{
    task = "quality_control"
    phase = 0
    status = "awaiting_gpu_measurement"
    limits_minutes = @{
        check = 3
        public = 20
        private = 40
    }
    measured = $null
    note = "Timing to be measured on GPU with Docker baseline image and full subset."
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
}
$runtime | ConvertTo-Json -Depth 5 | Set-Content -Path $RuntimeReport -Encoding UTF8

Write-Step "Done"
Write-Host "Subset:       $SubsetCsv"
Write-Host "Submission:   $SmokeOut"
Write-Host "Score report: $ScoreReport"
Write-Host "Runtime:      $RuntimeReport"
