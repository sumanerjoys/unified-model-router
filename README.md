# Unified Model Router

A production-ready API gateway that acts as a unified model router (OpenRouter-style). It exposes a single OpenAI-compatible `POST /v1/chat/completions` endpoint, translates the unified schema to real upstream LLM providers via the Adapter pattern, proxies live SSE streaming chunks back to the client, and performs silent, resilient fallback when a primary provider fails.

> Status: scaffolding (Stage 0 complete). Architecture, implementation, tests, CI, and deployment are being built in stages.

## Objective

- **Unified API & schema translation** — one standardized inference schema, dynamically routed to real providers.
- **Streaming proxy (SSE)** — chunked `text/event-stream` piped from upstream to client without buffering the full payload in memory.
- **Resilient fallback routing** — transient upstream errors (429/502/503, timeouts) transparently switch to a backup provider without client disruption.

## Planned architecture (layers)

```
API Layer        HTTP surface, auth guard, validation, SSE response
Router Layer     provider selection + fallback policy + deadlines
ProviderClient   transport: HTTP streaming, timeouts, error classification
Adapter Layer    PURE schema translation (unified <-> vendor), zero I/O
```

A full Mermaid architecture diagram and the Stream & Connection Management writeup will be added in the Design stage.

## Tech stack

- FastAPI + Uvicorn (async, native SSE via `StreamingResponse`)
- httpx (async streaming client, shared connection pool)
- Pydantic v2 + pydantic-settings
- pytest + pytest-asyncio + respx (hermetic tests)
- Docker + GitHub Actions (CI)

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in provider keys
uvicorn app.main:app --reload
```

## Repository

https://github.com/sumanerjoys/unified-model-router
