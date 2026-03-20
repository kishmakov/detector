from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Future: app.state.model = load_model()
    yield
    # Future: cleanup


app = FastAPI(lifespan=lifespan)

app.include_router(analyze.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
