import hashlib
import os
import re
import uuid
from collections import deque
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
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
MAX_PAGES = int(os.getenv("MAX_PAGES", "40"))

app = FastAPI(title="Open Firestarter", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
qdrant = QdrantClient(url=QDRANT_URL)


class IndexRequest(BaseModel):
    url: str
    limit: int = Field(default=10, ge=1, le=MAX_PAGES)


class ChatRequest(BaseModel):
    site_id: str
    question: str
    top_k: int = Field(default=5, ge=1, le=12)
    chat_model: str | None = None


class IndexedPage(BaseModel):
    url: str
    title: str | None = None
    chunks: int


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
    return url.rstrip("/")


def same_site(url: str, root: str) -> bool:
    return urlparse(url).netloc.replace("www.", "") == urlparse(root).netloc.replace("www.", "")


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, end)
    return chunks


async def ollama_embed(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama embedding failed: {response.text}")
    data = response.json()
    embedding = data.get("embedding")
    if not embedding:
        raise HTTPException(status_code=502, detail="Ollama returned no embedding. Did you pull nomic-embed-text?")
    return embedding


async def ollama_chat(question: str, context: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    system = (
        "You answer using only the provided website context. "
        "If the answer is not in the context, say you do not know from the indexed pages. "
        "Be concise and include source URLs when useful."
    )
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
        )
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


async def fetch_page(client: httpx.AsyncClient, url: str) -> tuple[str, str | None, list[str]] | None:
    try:
        response = await client.get(url, follow_redirects=True)
        ctype = response.headers.get("content-type", "")
        if response.status_code >= 400 or "text/html" not in ctype:
            return None
        html = response.text
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        links = []
        for a in soup.select("a[href]"):
            href = normalize_url(a.get("href", ""), base=url)
            if href:
                links.append(href)
        return text or "", title, links
    except Exception:
        return None


@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code < 400
    except Exception:
        pass
    return {
        "ok": True,
        "ollama_ok": ollama_ok,
        "ollama_base_url": OLLAMA_BASE_URL,
        "qdrant_url": QDRANT_URL,
        "chat_model": DEFAULT_CHAT_MODEL,
        "embed_model": DEFAULT_EMBED_MODEL,
    }


@app.post("/api/index")
async def index_site(req: IndexRequest) -> dict[str, Any]:
    root_url = normalize_url(req.url)
    if not root_url:
        raise HTTPException(status_code=400, detail="Invalid URL")

    sid = site_id_for(root_url)
    visited: set[str] = set()
    queue: deque[str] = deque([root_url])
    pages: list[IndexedPage] = []
    points: list[models.PointStruct] = []

    headers = {"User-Agent": "OpenFirestarterBot/0.1 (+local self-hosted crawler)"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        while queue and len(visited) < req.limit:
            url = queue.popleft()
            if url in visited or not same_site(url, root_url):
                continue
            visited.add(url)
            fetched = await fetch_page(client, url)
            if not fetched:
                continue
            text, title, links = fetched
            for link in links:
                if link not in visited and same_site(link, root_url) and len(queue) < req.limit * 4:
                    queue.append(link)
            chunks = chunk_text(text)
            if not chunks:
                continue

            page_chunk_count = 0
            for idx, chunk in enumerate(chunks):
                vector = await ollama_embed(chunk)
                await ensure_collection(len(vector))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sid}:{url}:{idx}"))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "site_id": sid,
                            "root_url": root_url,
                            "url": url,
                            "title": title,
                            "chunk": chunk,
                            "chunk_index": idx,
                        },
                    )
                )
                page_chunk_count += 1
            pages.append(IndexedPage(url=url, title=title, chunks=page_chunk_count))

    if not points:
        raise HTTPException(status_code=400, detail="No usable text was extracted. Try a different URL or lower restrictions.")

    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="site_id", match=models.MatchValue(value=sid))])
            ),
        )
    except Exception:
        pass

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"site_id": sid, "root_url": root_url, "pages_indexed": len(pages), "chunks_indexed": len(points), "pages": [p.model_dump() for p in pages]}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    question_vector = await ollama_embed(req.question)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,
        query_filter=models.Filter(must=[models.FieldCondition(key="site_id", match=models.MatchValue(value=req.site_id))]),
        limit=req.top_k,
        with_payload=True,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No indexed chunks found for that site_id")

    context_parts = []
    sources = []
    for hit in results:
        payload = hit.payload or {}
        url = payload.get("url", "")
        title = payload.get("title") or url
        chunk = payload.get("chunk", "")
        context_parts.append(f"Source: {title}\nURL: {url}\nText: {chunk}")
        sources.append({"url": url, "title": title, "score": hit.score})

    answer = await ollama_chat(req.question, "\n\n---\n\n".join(context_parts), model=req.chat_model or DEFAULT_CHAT_MODEL)
    return {"answer": answer, "sources": sources}
