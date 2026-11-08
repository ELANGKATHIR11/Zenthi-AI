# Native Windows Startup Script for Zenthi-AI OS
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "       Zenthi-AI OS - Local Native Startup       " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# 1. Check if Ollama is running
Write-Host "Checking Ollama connectivity on http://localhost:11434..." -ForegroundColor Yellow
try {
    $res = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
    Write-Host "[SUCCESS] Ollama server is running." -ForegroundColor Green
    $hasHybrid = $res.models | Where-Object { $_.name -like "*qwen3.5-hybrid*" }
    if ($hasHybrid) {
        Write-Host "[SUCCESS] Qwen 3.5 SLM+VLM Hybrid model is installed and ready." -ForegroundColor Green
    } else {
        Write-Host "[INFO] Building qwen3.5-hybrid:4b in Ollama..." -ForegroundColor Yellow
        ollama create qwen3.5-hybrid:4b -f ollama/Modelfile
    }
} catch {
    Write-Host "[WARNING] Ollama is not detected at http://localhost:11434." -ForegroundColor Red
    Write-Host "Please ensure Ollama is installed and running ('ollama serve') before processing queries." -ForegroundColor Yellow
}

# 2. Detect Python / Conda environment
$pythonCmd = "python"
$condaEnvPython = "C:\Users\elang\miniconda3\envs\dgpu-core\python.exe"
if (Test-Path $condaEnvPython) {
    $pythonCmd = $condaEnvPython
}

# 3. Launch Zenthi-AI backend & dashboard
Write-Host "`nStarting Zenthi-AI OS Dashboard on http://localhost:7860..." -ForegroundColor Green
& $pythonCmd app.py
