#!/usr/bin/env bash
set -euo pipefail

echo "Pulling embedding model: nomic-embed-text"
docker exec open-firestarter-ollama ollama pull nomic-embed-text

echo "Pulling chat model: llama3.1:8b"
docker exec open-firestarter-ollama ollama pull llama3.1:8b

echo "Done. If your machine is low on RAM, try: docker exec open-firestarter-ollama ollama pull qwen2.5:3b"
