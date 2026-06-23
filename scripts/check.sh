#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Checking health..."
curl -fsS "$BASE_URL/api/health" -o /tmp/open-firestarter-health.json
python -m json.tool /tmp/open-firestarter-health.json

printf "\nChecking scrape endpoint...\n"
curl -fsS -X POST "$BASE_URL/api/scrape" \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com"}' \
  -o /tmp/open-firestarter-scrape.json
python - <<'PY'
import json
payload=json.load(open('/tmp/open-firestarter-scrape.json'))
print({"url": payload.get("url"), "title": payload.get("title"), "markdown_chars": len(payload.get("markdown", "")), "links": len(payload.get("links", []))})
PY

printf "\nChecking crawl endpoint...\n"
curl -fsS -X POST "$BASE_URL/api/crawl" \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","limit":1,"max_depth":0}' \
  -o /tmp/open-firestarter-crawl.json
python - <<'PY'
import json
payload=json.load(open('/tmp/open-firestarter-crawl.json'))
print({"site_id": payload.get("site_id"), "pages_crawled": payload.get("pages_crawled")})
PY


printf "\nChecking export endpoint...\n"
curl -fsS -X POST "$BASE_URL/api/export" \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com","format":"csv","limit":1,"max_depth":0}' \
  -o /tmp/open-firestarter-export.csv
head -3 /tmp/open-firestarter-export.csv
