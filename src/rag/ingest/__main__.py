"""One-command ingestion: fetch Wikipedia corpus -> chunk -> build Qdrant hybrid index.

Usage: python -m rag.ingest [--limit N]
"""

import argparse

from rag.config import settings
from rag.ingest.articles import ARTICLES
from rag.ingest.chunk import chunk_article
from rag.ingest.fetch import fetch_corpus
from rag.ingest.index import build_index
from rag.store import client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only ingest first N articles")
    args = parser.parse_args()

    titles = ARTICLES[: args.limit] if args.limit else ARTICLES
    print(f"Fetching {len(titles)} Wikipedia articles (cache: data/corpus/)...")
    articles, missing = fetch_corpus(titles)
    if missing:
        print(f"WARNING: {len(missing)} titles not found: {missing}")

    cfg = settings()
    chunks = [
        c
        for a in articles
        for c in chunk_article(a, cfg.chunk_tokens, cfg.chunk_overlap_tokens)
    ]
    total_words = sum(len(c.text.split()) for c in chunks)
    print(f"Chunked {len(articles)} articles -> {len(chunks)} chunks (~{total_words:,} words)")

    print("Building Qdrant hybrid index (dense + BM25)...")
    build_index(chunks)
    print(f"Done. Collection '{cfg.collection}' at {cfg.qdrant_path}")
    client().close()


if __name__ == "__main__":
    main()
