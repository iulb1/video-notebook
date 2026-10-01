param(
    [string]$EnvironmentName = "video-notebook"
)

$ErrorActionPreference = "Stop"

$conda = Get-Command conda.exe -ErrorAction SilentlyContinue
if (-not $conda) {
    $fallback = "C:\ProgramData\miniconda3\Scripts\conda.exe"
    if (Test-Path -LiteralPath $fallback) {
        $conda = Get-Item -LiteralPath $fallback
    } else {
        throw "conda.exe was not found. Install Miniconda or Anaconda first."
    }
}

$condaPath = $conda.Source
$environmentRoot = Split-Path -Parent (Split-Path -Parent $condaPath)
$environmentPath = Join-Path $environmentRoot "envs\$EnvironmentName"
$environmentExists = Test-Path -LiteralPath (Join-Path $environmentPath "python.exe")

if (-not $environmentExists) {
    Write-Host "Creating Conda environment $EnvironmentName with Python 3.12..."
    & $condaPath create -n $EnvironmentName python=3.12 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }
}

Write-Host "Installing PyTorch for CUDA 12.8..."
& $condaPath run -n $EnvironmentName python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }

Write-Host "Installing project dependencies..."
& $condaPath run -n $EnvironmentName python -m pip install -r requirements.txt -r requirements-sensevoice.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

Write-Host "Running environment diagnostics..."
& $condaPath run -n $EnvironmentName python environment_check.py --backend sensevoice --device cuda
if ($LASTEXITCODE -ne 0) { throw "Environment diagnostics failed. Review the missing items above." }

Write-Host "Environment setup completed. Run the launcher batch file." -ForegroundColor Green
