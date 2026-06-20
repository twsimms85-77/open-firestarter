#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Checking Python syntax..."
python -m py_compile app/main.py

echo "Checking required files..."
test -f docker-compose.yml
test -f Dockerfile
test -f requirements.txt
test -f app/main.py
test -f static/index.html

echo "Checking Docker Compose config if Docker is available..."
if command -v docker >/dev/null 2>&1; then
  docker compose config >/dev/null
  echo "Docker Compose config OK"
else
  echo "Docker not installed here; skipping compose validation"
fi

echo "OK"
