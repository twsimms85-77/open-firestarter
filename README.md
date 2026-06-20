# Open Firestarter

Self-hosted Firestarter-style starter: crawl a website, extract clean text, embed it locally with Ollama, store vectors in Qdrant, and chat with the indexed site.

No hosted Firecrawl API key required.

## Fastest start

```bash
tar -xzf open-firestarter.tar.gz
cd open-firestarter
./scripts/start.sh
```

Then open:

```txt
http://localhost:8000
```

The first model pull can take a while. It downloads:

- `nomic-embed-text` for embeddings
- `llama3.1:8b` for chat

## If your computer has less RAM

Use Qwen 3B instead of Llama 8B:

```bash
docker exec open-firestarter-ollama ollama pull qwen2.5:3b
```

Edit `docker-compose.yml`:

```yaml
DEFAULT_CHAT_MODEL: qwen2.5:3b
```

Restart:

```bash
docker compose up -d
```

## What it runs

- FastAPI backend and web UI
- Ollama local models
- Qdrant vector database
- Trafilatura and BeautifulSoup crawler
- Local embeddings through Ollama
- Local chat through Ollama

## Check it

```bash
./scripts/check.sh
curl http://localhost:8000/api/health
```

Expected health response should show `ollama_ok: true` after Docker is up and Ollama is running.

## Use it

1. Open `http://localhost:8000`.
2. Enter a URL like `https://example.com`.
3. Keep page limit low first, like `5` or `10`.
4. Click Index site.
5. Ask questions with the generated `site_id`.

## API

Index:

```bash
curl -X POST http://localhost:8000/api/index \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","limit":5}'
```

Chat:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"site_id":"example-com-abc12345","question":"What does this site do?"}'
```

## Dev mode without Docker for the app

If you already have Ollama and Qdrant running separately:

```bash
./scripts/dev-no-docker.sh
```

## Known limits

This is the lean working version. JavaScript-heavy sites may need Playwright/Crawlee next. Captchas, bot blocks, and rate limits still exist with self-hosted crawling.

Next upgrade is Prometheus-style collector mode: describe the data you want, generate a plan, crawl, extract structured JSON/CSV.
