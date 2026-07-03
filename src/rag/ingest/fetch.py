"""Fetch plaintext article extracts from the English Wikipedia API, with a local cache.

Full-text extracts only come one page per request, so articles are fetched sequentially
with a polite delay and cached as JSON under data/corpus/ for reproducible re-runs.
"""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "production-rag-platform/1.0 (portfolio project; vireshkoli00@gmail.com)"
CORPUS_DIR = Path("data/corpus")


def _cache_path(title: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return CORPUS_DIR / f"{slug}.json"


def fetch_article(client: httpx.Client, title: str) -> dict | None:
    """Return {title, url, text} for one article, or None if it doesn't exist."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
        "titles": title,
    }
    resp = client.get(API_URL, params=params)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        return None
    return {
        "title": page["title"],
        "url": page["fullurl"],
        "text": page["extract"],
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def fetch_corpus(titles: list[str], delay_s: float = 0.15) -> tuple[list[dict], list[str]]:
    """Fetch all articles (cache-first). Returns (articles, missing_titles)."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    articles: list[dict] = []
    missing: list[str] = []
    seen_canonical: set[str] = set()

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for i, title in enumerate(titles):
            path = _cache_path(title)
            if path.exists():
                article = json.loads(path.read_text())
            else:
                article = fetch_article(client, title)
                if article is None:
                    missing.append(title)
                    continue
                path.write_text(json.dumps(article, ensure_ascii=False))
                time.sleep(delay_s)
            # redirects can collapse two list entries into one canonical page
            if article["title"] in seen_canonical:
                continue
            seen_canonical.add(article["title"])
            articles.append(article)
            if (i + 1) % 50 == 0:
                print(f"  fetched {i + 1}/{len(titles)} titles...")

    return articles, missing
