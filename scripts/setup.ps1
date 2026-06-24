# ==============================================================
# LexiFind — Project Setup Script (Windows PowerShell)
# Run this once before starting development.
# Usage: .\scripts\setup.ps1
# ==============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=================================================="
Write-Host "  LexiFind Setup"
Write-Host "=================================================="
Write-Host ""

# ── Step 1: Python dependencies ───────────────────────────────
Write-Host "[1/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "      OK Dependencies installed"
Write-Host ""

# ── Step 2: Download embedding model ─────────────────────────
Write-Host "[2/4] Downloading BGE-M3 embedding model (~570MB)..."
python -c @"
from FlagEmbedding import BGEM3FlagModel
print('      Pulling BAAI/bge-m3 from HuggingFace...')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print('      OK BGE-M3 ready')
"@
Write-Host ""

# ── Step 3: Download reranker model ──────────────────────────
Write-Host "[3/4] Downloading BGE-Reranker-v2-M3 (~570MB)..."
python -c @"
from FlagEmbedding import FlagReranker
print('      Pulling BAAI/bge-reranker-v2-m3 from HuggingFace...')
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
print('      OK Reranker ready')
"@
Write-Host ""

# ── Step 4: Verify .env exists ────────────────────────────────
Write-Host "[4/4] Checking .env file..."
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "      WARN .env created from .env.example"
    Write-Host "      Fill in GROQ_API_KEY and JWT_SECRET_KEY before running the app"
} else {
    Write-Host "      OK .env found"
}

Write-Host ""
Write-Host "=================================================="
Write-Host "  Setup complete! Next steps:"
Write-Host "  1. Fill in .env (GROQ_API_KEY, JWT_SECRET_KEY)"
Write-Host "  2. Start Qdrant:  docker compose up -d  (from WSL)"
Write-Host "  3. Start API:     uvicorn app.main:app --reload"
Write-Host "=================================================="
Write-Host ""