#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/twsimms85-77/open-firestarter.git}"
APP_DIR="${APP_DIR:-/opt/open-firestarter}"
CHAT_MODEL="${DEFAULT_CHAT_MODEL:-llama3.1:8b}"
EMBED_MODEL="${DEFAULT_EMBED_MODEL:-nomic-embed-text}"

echo "Open Firestarter VPS bootstrap"

if ! command -v git >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git curl ca-certificates
fi

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin is required. Installing via apt if available..."
  sudo apt-get update
  sudo apt-get install -y docker-compose-plugin
fi

sudo mkdir -p "$(dirname "$APP_DIR")"
if [ ! -d "$APP_DIR/.git" ]; then
  sudo git clone "$REPO_URL" "$APP_DIR"
else
  sudo git -C "$APP_DIR" pull --ff-only
fi
sudo chown -R "$USER":"$USER" "$APP_DIR"
cd "$APP_DIR"

mkdir -p data
docker compose up -d --build

echo "Pulling Ollama models. This can take several minutes..."
docker exec open-firestarter-ollama ollama pull "$EMBED_MODEL"
docker exec open-firestarter-ollama ollama pull "$CHAT_MODEL"

echo "Waiting for API..."
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/api/health >/tmp/open-firestarter-health.json 2>/dev/null; then
    cat /tmp/open-firestarter-health.json
    echo
    break
  fi
  sleep 2
  if [ "$i" = "60" ]; then
    echo "API did not become healthy in time. Check: docker compose logs -f"
    exit 1
  fi
done

BASE_URL=http://localhost:8000 ./scripts/check.sh

echo "Done. Open http://YOUR_SERVER_IP:8000"
