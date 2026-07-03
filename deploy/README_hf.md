---
title: Production RAG Platform
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Production RAG Platform

Hybrid retrieval + cross-encoder reranking + an agentic groundedness self-check over
311 Wikipedia AI/ML articles. Ask a question at `/`, watch live cost/latency at `/dashboard`.

Source, architecture, and measured evaluation results:
**https://github.com/vireshkoli/production-rag-platform**

Set the `ANTHROPIC_API_KEY` secret in this Space's settings for answering to work.
