This is a project for Google Chrome extension which analyzes the text.

## Project Structure

The `server/` directory contains a Python + FastAPI backend that provides text analysis endpoints.

### `server/`

- **Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2
- **Venv:** `server/.venv/` — activate with `source .venv/bin/activate` or run directly via `.venv/bin/pytest`, `.venv/bin/uvicorn`
- **Run locally:** `docker-compose up` (hot reload) or `uvicorn app.main:app --reload` inside the venv
- **Run tests:** `.venv/bin/pytest tests/ -v`
- **Deploy:** `bash scripts/deploy.sh` — rsyncs to remote host aliased `GC` in `~/.ssh/config`, builds Docker image on remote, manages via systemd unit

#### Layout

```
server/
├── app/
│   ├── main.py              # FastAPI app, lifespan hook, GET /health
│   ├── models.py            # Pydantic schemas: AnalyzeRequest, AnalyzeResponse
│   ├── routers/analyze.py   # POST /analyze
│   └── services/
│       └── text_analysis.py # count_words() — pure business logic
├── tests/
│   ├── conftest.py          # Session-scoped TestClient fixture
│   └── test_analyze.py      # Unit + integration tests
├── scripts/deploy.sh        # Single-command remote deploy
├── Dockerfile
├── docker-compose.yml       # Local dev only
└── requirements.txt
```

#### API

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Returns `{"status": "ok"}`           |
| POST   | `/analyze` | Accepts `{"text": "..."}`, returns `{"word_count": N}` |

#### GPU Upgrade Path

Swap `FROM` in `Dockerfile` to `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`, add `--gpus all` to the systemd unit's `docker run` command, then load the model in the `lifespan` hook (`app.state.model`).
