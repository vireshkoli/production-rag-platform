"""Build the Qdrant hybrid index (dense + BM25 sparse vectors) from chunks."""

from qdrant_client import models

from rag.config import settings
from rag.embeddings import dense_encoder, embed_passages, embed_sparse
from rag.ingest.chunk import Chunk
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, client, recreate_collection

BATCH = 128


def build_index(chunks: list[Chunk]) -> None:
    enc = dense_encoder()
    dim = getattr(enc, "get_embedding_dimension", enc.get_sentence_embedding_dimension)()
    recreate_collection(dim)

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        texts = [c.text for c in batch]
        dense = embed_passages(texts)
        sparse = embed_sparse(texts)
        client().upsert(
            collection_name=settings().collection,
            points=[
                models.PointStruct(
                    id=c.id,
                    vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv},
                    payload={
                        "title": c.title,
                        "section": c.section,
                        "text": c.text,
                        "url": c.url,
                    },
                )
                for c, dv, sv in zip(batch, dense, sparse, strict=True)
            ],
        )
        print(f"  indexed {min(start + BATCH, len(chunks))}/{len(chunks)} chunks...")
