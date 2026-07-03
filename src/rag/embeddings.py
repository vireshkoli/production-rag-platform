"""Lazy-loaded local models shared by ingestion and query-time retrieval.

Dense: BAAI/bge-small-en-v1.5 (384-d). BGE convention: passages are embedded raw,
queries get an instruction prefix. Sparse: Qdrant/bm25 term-frequency vectors (IDF is
applied server-side by Qdrant). Reranker: ms-marco MiniLM cross-encoder.
"""

from functools import lru_cache

from qdrant_client import models

from rag.config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def dense_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings().dense_model)


@lru_cache
def sparse_encoder():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(settings().sparse_model)


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings().reranker_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return dense_encoder().encode(texts, normalize_embeddings=True, batch_size=64).tolist()


def embed_query_dense(query: str) -> list[float]:
    return dense_encoder().encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in sparse_encoder().embed(texts)
    ]


def warm_up() -> None:
    """Load all models eagerly (used at server startup so first query isn't slow)."""
    embed_passages(["warm up"])
    embed_query_dense("warm up")
    embed_sparse(["warm up"])
    reranker().predict([("warm up", "warm up")])
