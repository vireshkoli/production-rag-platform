"""Query-time retrieval.

Full pipeline: dense + BM25 prefetch fused server-side with Reciprocal Rank Fusion,
then a cross-encoder reranks the fused candidates and the top_k survive.
Baseline pipeline (for the eval comparison): dense-only top_k, no fusion, no reranker.
"""

from dataclasses import dataclass

from qdrant_client import models

from rag.config import settings
from rag.embeddings import embed_query_dense, embed_sparse, reranker
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, client


@dataclass
class RetrievedChunk:
    id: str
    title: str
    section: str
    text: str
    url: str
    score: float

    @classmethod
    def from_point(cls, point) -> "RetrievedChunk":
        p = point.payload
        return cls(
            id=str(point.id),
            title=p["title"],
            section=p["section"],
            text=p["text"],
            url=p["url"],
            score=point.score,
        )


def dense_search(query: str, limit: int) -> list[RetrievedChunk]:
    """Baseline: plain dense similarity search."""
    result = client().query_points(
        collection_name=settings().collection,
        query=embed_query_dense(query),
        using=DENSE_VECTOR,
        limit=limit,
        with_payload=True,
    )
    return [RetrievedChunk.from_point(p) for p in result.points]


def hybrid_search(query: str, limit: int) -> list[RetrievedChunk]:
    """Dense + BM25 prefetch, fused with RRF inside Qdrant."""
    cfg = settings()
    result = client().query_points(
        collection_name=cfg.collection,
        prefetch=[
            models.Prefetch(
                query=embed_query_dense(query), using=DENSE_VECTOR, limit=cfg.prefetch_k
            ),
            models.Prefetch(
                query=embed_sparse([query])[0], using=SPARSE_VECTOR, limit=cfg.prefetch_k
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [RetrievedChunk.from_point(p) for p in result.points]


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Cross-encoder rescoring; returns the top_k chunks by relevance to the query."""
    if not chunks:
        return []
    scores = reranker().predict([(query, c.text) for c in chunks])
    scored = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [
        RetrievedChunk(**{**c.__dict__, "score": float(s)}) for c, s in scored[:top_k]
    ]


def retrieve(query: str, pipeline: str = "full") -> list[RetrievedChunk]:
    """Entry point used by the answer pipelines.

    pipeline="baseline": dense-only top_k (what a naive RAG system does).
    pipeline="full":     hybrid RRF over rerank_candidates, cross-encoder -> top_k.
    """
    cfg = settings()
    if pipeline == "baseline":
        return dense_search(query, cfg.top_k)
    candidates = hybrid_search(query, cfg.rerank_candidates)
    return rerank(query, candidates, cfg.top_k)
