# RAG Pipeline

![CI](https://github.com/charlesdc003/rag-pipeline/actions/workflows/ci.yml/badge.svg)

A retrieval-augmented generation pipeline using local LLMs, pgvector, and W&B Weave tracing.

## What it does

Accepts a natural language query, retrieves relevant document chunks via semantic similarity search, and generates a grounded response with confidence scoring and action routing.

## Stack

- FastAPI + Pydantic v2
- PostgreSQL + pgvector (vector similarity search)
- Ollama (local embeddings + generation on RTX 5080)
- W&B Weave (trace logging)
- pytest + GitHub Actions CI

## Architecture

Query → nomic-embed-text embeddings → pgvector similarity search → llama3.2 generation → structured JSON response

## Performance

- Embedding model: nomic-embed-text (768 dimensions)
- Generation model: llama3.2 (local, no API cost)
- Average latency: ~5.7s end to end
- Confidence routing: auto_reply > 0.7, escalate < 0.4, review otherwise

## Run locally

```bash
docker compose up -d
uv sync
uv run uvicorn main:app --reload
```

## Limitations

- Scoring service runs deterministic rules only — no LLM reranking of retrieved chunks
- No authentication on the API endpoint
- Knowledge base must be manually ingested via ingest_document()
- Latency is hardware dependent — benchmarked on RTX 5080