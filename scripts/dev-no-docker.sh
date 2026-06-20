#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
export DEFAULT_CHAT_MODEL="${DEFAULT_CHAT_MODEL:-llama3.1:8b}"
export DEFAULT_EMBED_MODEL="${DEFAULT_EMBED_MODEL:-nomic-embed-text}"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
