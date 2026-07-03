"""Qdrant access. Embedded local mode by default; set RAG_QDRANT_URL for a server."""

import os
from functools import lru_cache

from qdrant_client import QdrantClient, models

from rag.config import settings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


@lru_cache
def client() -> QdrantClient:
    url = os.environ.get("RAG_QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    settings().qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings().qdrant_path))


def recreate_collection(dense_dim: int) -> None:
    name = settings().collection
    if client().collection_exists(name):
        client().delete_collection(name)
    client().create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
