# Open Firestarter

Self-hosted Firecrawl/Prometheus-style starter: scrape pages into clean Markdown, crawl sites, extract structured data, index website chunks into Qdrant, and chat with indexed sites using local Ollama models.

No hosted Firecrawl API key required.

## What v0.2 does

- `/api/scrape` — single page to Markdown/text/links/metadata.
- `/api/crawl` — same-site crawling with page limit, max depth, include/exclude regex filters, and Markdown output.
- `/api/index` — crawl + local embeddings via Ollama + deterministic UUIDv5 point IDs in Qdrant.
- `/api/chat` — RAG chat over indexed website chunks.
- `/api/extract` — Prometheus-style plain-English extraction into JSON using the local chat model.
- `/api/jobs` — simple background crawl/index job wrapper.
- Mobile-friendly UI at `/`.

## Quick start

```bash
git clone https://github.com/twsimms85-77/open-firestarter.git
cd open-firestarter
./scripts/start.sh
```

Open:

```txt
http://localhost:8000
```

The first model pull can take a while. It downloads:

- `nomic-embed-text` for embeddings
- `llama3.1:8b` for chat

## Lower-RAM option

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

## Health check

```bash
curl http://localhost:8000/api/health
```

Expected when fully running:

```json
{
  "ok": true,
  "ollama_ok": true,
  "qdrant_ok": true
}
```

## API examples

Scrape one page:

```bash
curl -X POST http://localhost:8000/api/scrape \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com"}'
```

Crawl a site:

```bash
curl -X POST http://localhost:8000/api/crawl \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","limit":5,"max_depth":2}'
```

Index a site:

```bash
curl -X POST http://localhost:8000/api/index \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","limit":5,"max_depth":2}'
```

Chat:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"site_id":"example-com-abc12345","question":"What does this site do?"}'
```

Extract structured JSON:

```bash
curl -X POST http://localhost:8000/api/extract \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","instruction":"Extract the organization name, summary, links, and source_url."}'
```

## Dev mode without Docker for the app

If Ollama and Qdrant are already running separately:

```bash
./scripts/dev-no-docker.sh
```

## Known limits versus Firecrawl quality

This is now a serious local foundation, but Firecrawl-level production crawling still needs more work:

- JavaScript rendering via Playwright/Crawlee or Browserless.
- Robots.txt/sitemap policy controls.
- Queue persistence with Redis/Postgres instead of in-memory jobs.
- Proxy rotation and anti-bot handling for difficult public sites.
- Better LLM extraction validation/retries.
- Authenticated crawling/session capture.
- Exports to CSV/JSONL and webhooks.

The design keeps those optional so the base stack stays cheap and self-hosted.
