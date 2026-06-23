import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "llama3.1:8b")
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "firestarter_chunks")
MAX_PAGES = int(os.getenv("MAX_PAGES", "80"))
CRAWL_DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "0.25"))
USER_AGENT = os.getenv("USER_AGENT", "OpenFirestarterBot/0.2 (+self-hosted crawler)")

app = FastAPI(title="Open Firestarter", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
qdrant = QdrantClient(url=QDRANT_URL)

JobStatus = Literal["queued", "running", "done", "error"]


class ScrapeRequest(BaseModel):
    url: str
    include_html: bool = False
    only_main_content: bool = True


class CrawlRequest(BaseModel):
    url: str
    limit: int = Field(default=10, ge=1, le=MAX_PAGES)
    max_depth: int = Field(default=2, ge=0, le=8)
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    allow_external: bool = False
    index: bool = True


class IndexRequest(CrawlRequest):
    pass


class ChatRequest(BaseModel):
    site_id: str
    question: str
    top_k: int = Field(default=6, ge=1, le=20)
    chat_model: str | None = None


class ExtractRequest(BaseModel):
    url: str
    instruction: str = Field(..., description="Plain-English extraction instruction")
    limit: int = Field(default=5, ge=1, le=MAX_PAGES)
    max_depth: int = Field(default=1, ge=0, le=5)
    json_schema_hint: dict[str, Any] | None = None
    chat_model: str | None = None


class JobRequest(BaseModel):
    kind: Literal["crawl", "index"] = "index"
    crawl: CrawlRequest


@dataclass
class ExtractedPage:
    url: str
    title: str | None
    description: str | None
    markdown: str
    text: str
    links: list[str]
    status_code: int
    content_type: str
    elapsed_ms: int


jobs: dict[str, dict[str, Any]] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def site_id_for(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    stamp = hashlib.sha1(url.encode()).hexdigest()[:8]
    return re.sub(r"[^a-zA-Z0-9_-]", "-", f"{host}-{stamp}")


def normalize_url(url: str, base: str | None = None) -> str | None:
    if base:
        url = urljoin(base, url)
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    clean_path = parsed.path.rstrip("/") or ""
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme}://{parsed.netloc}{clean_path}{query}"


def same_site(url: str, root: str) -> bool:
    return urlparse(url).netloc.replace("www.", "") == urlparse(root).netloc.replace("www.", "")


def pattern_allowed(url: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    if include_patterns and not any(re.search(p, url) for p in include_patterns):
        return False
    if exclude_patterns and any(re.search(p, url) for p in exclude_patterns):
        return False
    return True


def html_to_markdown(html: str, url: str, only_main_content: bool = True) -> str:
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_precision=only_main_content,
    )
    if markdown and len(markdown.strip()) > 80:
        return markdown.strip()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    target = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    for el in target.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "td", "th"]):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if el.name == "h1":
            lines.append(f"# {text}")
        elif el.name == "h2":
            lines.append(f"## {text}")
        elif el.name == "h3":
            lines.append(f"### {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines).strip()


def markdown_to_plain_text(markdown: str) -> str:
    text = re.sub(r"[`*_>#\[\]()]", " ", markdown)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 220) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_break = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if sentence_break > start + 500:
                end = sentence_break + 1
        chunk = text[start:end].strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, end)
    return chunks


async def ollama_embed(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json={"model": model, "prompt": text})
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama embedding failed: {response.text}")
    embedding = response.json().get("embedding")
    if not embedding:
        raise HTTPException(status_code=502, detail=f"Ollama returned no embedding. Pull {model} first.")
    return embedding


async def ollama_chat(messages: list[dict[str, str]], model: str = DEFAULT_CHAT_MODEL) -> str:
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json={"model": model, "stream": False, "messages": messages})
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama chat failed: {response.text}")
    return response.json().get("message", {}).get("content", "")


async def ensure_collection(vector_size: int) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


async def fetch_page(client: httpx.AsyncClient, url: str, only_main_content: bool = True) -> ExtractedPage | None:
    started = now_ms()
    try:
        response = await client.get(url, follow_redirects=True)
        ctype = response.headers.get("content-type", "")
        if response.status_code >= 400 or "text/html" not in ctype:
            return None
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        desc_tag = soup.select_one('meta[name="description"], meta[property="og:description"]')
        description = desc_tag.get("content", "").strip() if desc_tag else None
        links: list[str] = []
        seen: set[str] = set()
        for a in soup.select("a[href]"):
            href = normalize_url(a.get("href", ""), base=str(response.url))
            if href and href not in seen:
                seen.add(href)
                links.append(href)
        markdown = html_to_markdown(html, str(response.url), only_main_content=only_main_content)
        text = markdown_to_plain_text(markdown)
        return ExtractedPage(
            url=str(response.url),
            title=title,
            description=description,
            markdown=markdown,
            text=text,
            links=links,
            status_code=response.status_code,
            content_type=ctype,
            elapsed_ms=now_ms() - started,
        )
    except Exception:
        return None


async def crawl_pages(req: CrawlRequest) -> dict[str, Any]:
    root_url = normalize_url(req.url)
    if not root_url:
        raise HTTPException(status_code=400, detail="Invalid URL")

    sid = site_id_for(root_url)
    visited: set[str] = set()
    queued: set[str] = {root_url}
    queue: deque[tuple[str, int]] = deque([(root_url, 0)])
    pages: list[ExtractedPage] = []
    skipped: list[dict[str, Any]] = []

    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    async with httpx.AsyncClient(timeout=35, headers=headers) as client:
        while queue and len(visited) < req.limit:
            url, depth = queue.popleft()
            if url in visited:
                continue
            if not req.allow_external and not same_site(url, root_url):
                skipped.append({"url": url, "reason": "external"})
                continue
            if not pattern_allowed(url, req.include_patterns, req.exclude_patterns):
                skipped.append({"url": url, "reason": "pattern"})
                continue

            visited.add(url)
            page = await fetch_page(client, url)
            if not page or not page.text:
                skipped.append({"url": url, "reason": "no_text_or_fetch_failed"})
                continue
            pages.append(page)

            if depth < req.max_depth:
                for link in page.links:
                    if link in visited or link in queued:
                        continue
                    if not req.allow_external and not same_site(link, root_url):
                        continue
                    if not pattern_allowed(link, req.include_patterns, req.exclude_patterns):
                        continue
                    queued.add(link)
                    queue.append((link, depth + 1))
            if CRAWL_DELAY_SECONDS > 0:
                await asyncio.sleep(CRAWL_DELAY_SECONDS)

    return {
        "site_id": sid,
        "root_url": root_url,
        "pages_crawled": len(pages),
        "skipped": skipped[:50],
        "pages": [
            {
                "url": p.url,
                "title": p.title,
                "description": p.description,
                "markdown": p.markdown,
                "text_chars": len(p.text),
                "links_found": len(p.links),
                "elapsed_ms": p.elapsed_ms,
            }
            for p in pages
        ],
    }


async def index_crawl(req: CrawlRequest) -> dict[str, Any]:
    crawl = await crawl_pages(req)
    points: list[models.PointStruct] = []
    total_chunks = 0

    for page in crawl["pages"]:
        chunks = chunk_text(markdown_to_plain_text(page["markdown"]))
        for idx, chunk in enumerate(chunks):
            vector = await ollama_embed(chunk)
            await ensure_collection(len(vector))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{crawl['site_id']}:{page['url']}:{idx}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "site_id": crawl["site_id"],
                        "root_url": crawl["root_url"],
                        "url": page["url"],
                        "title": page["title"],
                        "description": page["description"],
                        "chunk": chunk,
                        "chunk_index": idx,
                    },
                )
            )
            total_chunks += 1

    if not points:
        raise HTTPException(status_code=400, detail="No usable text was extracted/indexed.")

    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="site_id", match=models.MatchValue(value=crawl["site_id"]))])
            ),
        )
    except Exception:
        pass

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    crawl["chunks_indexed"] = total_chunks
    crawl.pop("pages", None)
    return crawl


@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ollama_ok = False
    qdrant_ok = False
    models_available: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code < 400
            if ollama_ok:
                models_available = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        pass
    return {
        "ok": True,
        "ollama_ok": ollama_ok,
        "qdrant_ok": qdrant_ok,
        "ollama_base_url": OLLAMA_BASE_URL,
        "qdrant_url": QDRANT_URL,
        "chat_model": DEFAULT_CHAT_MODEL,
        "embed_model": DEFAULT_EMBED_MODEL,
        "models_available": models_available,
        "version": "0.2.0",
    }


@app.post("/api/scrape")
async def scrape(req: ScrapeRequest) -> dict[str, Any]:
    url = normalize_url(req.url)
    if not url:
        raise HTTPException(status_code=400, detail="Invalid URL")
    async with httpx.AsyncClient(timeout=35, headers={"User-Agent": USER_AGENT}) as client:
        page = await fetch_page(client, url, only_main_content=req.only_main_content)
    if not page:
        raise HTTPException(status_code=400, detail="Page could not be fetched or had no readable HTML.")
    data = {
        "url": page.url,
        "title": page.title,
        "description": page.description,
        "markdown": page.markdown,
        "text": page.text,
        "links": page.links,
        "status_code": page.status_code,
        "content_type": page.content_type,
        "elapsed_ms": page.elapsed_ms,
    }
    if req.include_html:
        async with httpx.AsyncClient(timeout=35, headers={"User-Agent": USER_AGENT}) as client:
            html_response = await client.get(url, follow_redirects=True)
            data["html"] = html_response.text
    return data


@app.post("/api/crawl")
async def crawl(req: CrawlRequest) -> dict[str, Any]:
    return await crawl_pages(req)


@app.post("/api/index")
async def index_site(req: IndexRequest) -> dict[str, Any]:
    return await index_crawl(req)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    question_vector = await ollama_embed(req.question)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,
        query_filter=models.Filter(must=[models.FieldCondition(key="site_id", match=models.MatchValue(value=req.site_id))]),
        limit=req.top_k,
    )
    context_parts = []
    sources = []
    for result in results:
        payload = result.payload or {}
        context_parts.append(f"Source: {payload.get('url')}\nTitle: {payload.get('title')}\n{payload.get('chunk')}")
        sources.append({"url": payload.get("url"), "title": payload.get("title"), "score": result.score})
    if not context_parts:
        raise HTTPException(status_code=404, detail="No indexed chunks found for that site_id.")
    system = (
        "You answer using only the provided website context. If the answer is not in the context, say you do not know from the indexed pages. "
        "Be concise, factual, and cite source URLs when useful."
    )
    joined_context = "\n\n---\n\n".join(context_parts)
    answer = await ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{joined_context}\n\nQuestion: {req.question}"},
        ],
        model=req.chat_model or DEFAULT_CHAT_MODEL,
    )
    return {"answer": answer, "sources": sources}


@app.post("/api/extract")
async def extract(req: ExtractRequest) -> dict[str, Any]:
    crawl_req = CrawlRequest(url=req.url, limit=req.limit, max_depth=req.max_depth, index=False)
    crawled = await crawl_pages(crawl_req)
    combined = "\n\n---PAGE---\n\n".join(
        f"URL: {p['url']}\nTITLE: {p['title']}\nMARKDOWN:\n{p['markdown'][:8000]}" for p in crawled["pages"]
    )
    schema_hint = json.dumps(req.json_schema_hint or {}, indent=2)
    system = (
        "You are a web data extraction engine. Return only valid JSON. No markdown fences. "
        "If a value is not present, use null. Include source_url fields when possible."
    )
    user = f"Instruction: {req.instruction}\n\nJSON schema hint, if any:\n{schema_hint}\n\nWebsite content:\n{combined[:30000]}"
    raw = await ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=req.chat_model or DEFAULT_CHAT_MODEL,
    )
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"raw": raw}
    return {"site_id": crawled["site_id"], "pages_crawled": crawled["pages_crawled"], "data": parsed}


async def run_job(job_id: str, req: JobRequest) -> None:
    jobs[job_id].update({"status": "running", "started_at": time.time()})
    try:
        result = await index_crawl(req.crawl) if req.kind == "index" else await crawl_pages(req.crawl)
        jobs[job_id].update({"status": "done", "finished_at": time.time(), "result": result})
    except Exception as exc:
        jobs[job_id].update({"status": "error", "finished_at": time.time(), "error": str(exc)})


@app.post("/api/jobs")
async def create_job(req: JobRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "status": "queued", "kind": req.kind, "created_at": time.time()}
    background_tasks.add_task(run_job, job_id, req)
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
