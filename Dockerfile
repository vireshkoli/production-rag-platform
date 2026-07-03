# Multi-stage build. Models and the Qdrant index are baked into the image so the
# container is self-contained and the first query is fast.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    RAG_QDRANT_PATH=/app/data/qdrant \
    RAG_TELEMETRY_DB=/app/data/telemetry.sqlite3

WORKDIR /app

# --- deps (cached layer) ---
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# --- bake local models into the image (no download at runtime) ---
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" && \
python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"

# --- build the Qdrant index at image-build time (corpus cache is committed) ---
COPY evals ./evals
COPY data/corpus ./data/corpus
RUN python -m rag.ingest

COPY static ./static

# HF Spaces sends traffic to port 7860
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn rag.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
