#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Starting Open Firestarter, Qdrant, and Ollama..."
docker compose up -d --build

echo "Pulling local Ollama models..."
./scripts/pull-models.sh

echo "Ready: http://localhost:8000"
