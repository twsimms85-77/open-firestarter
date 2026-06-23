# Open Firestarter

Self-hosted Firecrawl/Prometheus-style starter: scrape pages into clean Markdown, crawl sites, extract structured data, index website chunks into Qdrant, and chat with indexed sites using local Ollama models.

No hosted Firecrawl API key required.

## What v0.4 does

- `/api/scrape` — single page to Markdown/text/links/metadata, with retries and optional JavaScript rendering.
- `/api/batch/scrape` — scrape up to 100 URLs in one request.
- `/api/map` — discover URLs from sitemap plus link traversal without extracting full page content.
- `/api/crawl` — same-site crawling with page limit, max depth, include/exclude regex filters, robots.txt checks, sitemap seeding, retries, and Markdown output.
- `/api/export` — crawl output as JSON, JSONL, or CSV.
- `/api/index` — crawl + local embeddings via Ollama + deterministic UUIDv5 point IDs in Qdrant.
- `/api/chat` — RAG chat over indexed website chunks.
- `/api/extract` — Prometheus-style plain-English extraction into JSON using the local chat model.
- `/api/jobs` — persistent background crawl/index/extract jobs saved in `data/jobs.json`.
- Optional Playwright Chromium rendering when `ENABLE_JS_RENDERING=true`.
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

Covered by v0.4: map endpoint, batch scrape, sitemap discovery, robots.txt handling, retries, persistent job state, CSV/JSONL exports, optional Playwright rendering, and one-command VPS bootstrap. Previously covered in v0.3: sitemap discovery, robots.txt handling, retries, persistent job state, CSV/JSONL exports, and optional Playwright rendering.


This is now a serious local foundation, but Firecrawl-level production crawling still needs more work:

- Distributed queues with Redis/Postgres for very large crawls.
- Proxy rotation and anti-bot handling for difficult public sites.
- Better LLM extraction validation/retries.
- Authenticated crawling/session capture.
- Exports to CSV/JSONL and webhooks.

The design keeps those optional so the base stack stays cheap and self-hosted.


## Export examples

CSV export:

```bash
curl -X POST http://localhost:8000/api/export \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","format":"csv","limit":5}' \
  -o crawl.csv
```

JavaScript-rendered scrape:

```bash
curl -X POST http://localhost:8000/api/scrape \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","render_js":true}'
```


## One-command VPS bootstrap

On a fresh Ubuntu VPS:

```bash
curl -fsSL https://raw.githubusercontent.com/twsimms85-77/open-firestarter/main/scripts/bootstrap-vps.sh | bash
```

Then open:

```txt
http://YOUR_SERVER_IP:8000
```

This installs Docker if needed, clones/updates the repo, starts the stack, pulls Ollama models, and runs the check script.
