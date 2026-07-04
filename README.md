# Unified Model Router

A production-ready API gateway that acts as a unified model router (OpenRouter-style). It exposes a single OpenAI-compatible `POST /v1/chat/completions` endpoint, translates the unified schema to real upstream LLM providers via the Adapter pattern, proxies live SSE streaming chunks back to the client, and performs silent, resilient fallback when a primary provider fails.

![CI](https://github.com/sumanerjoys/unified-model-router/actions/workflows/ci.yml/badge.svg)

## Objective

- **Unified API & schema translation** — one standardized inference schema, dynamically routed to real providers.
- **Streaming proxy (SSE)** — chunked `text/event-stream` piped from upstream to client without buffering the full payload in memory.
- **Resilient fallback routing** — transient upstream errors (429/502/503, timeouts) transparently switch to a backup provider without client disruption.

## Architecture (layers)

```
API Layer        HTTP surface, auth guard, validation, SSE response
Router Layer     provider selection + fallback policy + deadlines
ProviderClient   transport: HTTP streaming, timeouts, error classification
Adapter Layer    PURE schema translation (unified <-> vendor), zero I/O
```

See **[DESIGN.md](./DESIGN.md)** for the full problem statement, architecture diagrams (Mermaid), unified schema, error taxonomy, fallback policy, and stream/connection management.

## Tech stack

- FastAPI + Uvicorn (async, native SSE via `StreamingResponse`)
- httpx (async streaming client, shared connection pool)
- Pydantic v2 + pydantic-settings
- pytest + pytest-asyncio + respx (hermetic tests)
- Docker + GitHub Actions (CI)

## Project layout

```
app/
  main.py            # app factory + lifespan (shared AsyncClient) + health/ready
  config.py          # env-driven settings
  api/routes.py      # POST /v1/chat/completions (SSE + non-streaming)
  core/
    router.py        # provider selection + bounded silent fallback
    provider_client.py  # transport: streaming, timeouts, error classification
    errors.py        # error taxonomy (TRANSIENT / TIMEOUT / FATAL)
  adapters/          # PURE schema translation (openai, mock) + registry
  models/unified.py  # unified request/response/chunk schema
mock_provider/       # local mock LLM provider (deterministic; forced-failure modes)
tests/               # unit + integration (40 tests)
```

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in provider keys
```

### Configuration

All settings load from environment / `.env` (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `PRIMARY_BASE_URL` / `PRIMARY_API_KEY` | Real OpenAI-compatible provider | `https://api.openai.com/v1` |
| `FALLBACK_BASE_URL` / `FALLBACK_API_KEY` | Backup provider (defaults to local mock) | `http://localhost:9100/v1` |
| `MAX_FALLBACK_HOPS` | Max provider switches per request | `2` |
| `REQUEST_DEADLINE_SECONDS` | Overall wall-clock deadline | `60` |
| `UPSTREAM_CONNECT_TIMEOUT` / `UPSTREAM_READ_TIMEOUT` | Per-attempt timeouts | `5` / `60` |
| `GATEWAY_API_KEYS` | Comma-separated allowed gateway keys (extension; empty disables) | _(empty)_ |

## Running

### 1. Start the local mock provider (optional, for the fallback demo)

```bash
uvicorn mock_provider.server:app --port 9100
```

The mock provider speaks a deliberately non-OpenAI schema (to exercise real
adapter translation) and supports forced failure modes for demos/tests:

```bash
# Force a 429 (or 503, 500, 404, timeout) to trigger the gateway's fallback:
curl -s -X POST "http://localhost:9100/v1/generate?fail=429" -d '{"stream":true}'
```

### 2. Start the gateway

```bash
uvicorn app.main:app --reload --port 8000
```

## Sample requests

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in 3 words."}],
    "stream": true
  }'
```

Example response stream:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Non-streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Demonstrating silent fallback

Point `PRIMARY_BASE_URL` at a failing endpoint (e.g. the mock with `?fail=429`)
and `FALLBACK_BASE_URL` at a healthy provider. The client still receives a clean
`200` stream — the upstream failure is invisible. A correlation id is returned in
the `X-Request-ID` response header, and the fallback chain is logged server-side.

## Tests

```bash
pytest -q            # 40 tests, ~1.3s
pytest --durations=5 # show slowest
```

- **Unit** (`tests/unit`): adapter translation, chunk parsing, error classification — pure, zero mocking.
- **Integration** (`tests/integration`): happy streaming, 429/503 silent fallback, all-providers-fail, FATAL no-fallback, bounded hops, timeout→fallback, client-disconnect cleanup, request validation.

Tests are hermetic: upstream is served in-process by the mock provider via
httpx `ASGITransport`, so no real network or LLM keys are needed. **Caveat:**
because `ASGITransport` has no real socket, tests do not exercise real socket
timeouts/TLS/connection resets; the timeout path is verified by injecting
`httpx.ReadTimeout` at the transport layer (equivalent to what a real socket
timeout raises).

## Docker

```bash
docker build -t unified-model-router .
docker run --rm -p 8000:8000 --env-file .env unified-model-router
```

The image uses `python:3.14-slim`, installs pinned dependencies, runs as a
non-root user, and includes a container `HEALTHCHECK` on `/health`.

## Deployment (DigitalOcean App Platform)

The service is deployed live on DigitalOcean App Platform, built from the
`Dockerfile` in this repo. The spec lives at [`.do/app.yaml`](./.do/app.yaml).

```bash
# One-time: authorize GitHub for App Platform in the DO console, then:
doctl apps create --spec .do/app.yaml
# Set the provider key as a secret (never committed):
doctl apps update <APP_ID> --spec .do/app.yaml   # with PRIMARY_API_KEY provided
```

Verified live (streaming through the deployed gateway to DigitalOcean Serverless
Inference `gpt-oss-120b`):

```bash
curl -N -X POST "$APP_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-gpt-oss-120b","messages":[{"role":"user","content":"Say hello in 3 words."}],"stream":true}'
# -> streamed unified SSE chunks + data: [DONE]
```

`GET /health` and `GET /ready` confirm liveness/readiness on the deployed URL.
`deploy_on_push: true` redeploys automatically on every push to `main`.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs the full test suite on every
push and pull request (Python 3.14).

## Repository

https://github.com/sumanerjoys/unified-model-router
