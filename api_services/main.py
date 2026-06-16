"""
main.py
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
# Right - split into correct modules
from src.monitoring.langsmith_tracer import setup_langsmith
from src.monitoring.metrices import start_metrics_server
from api_services.routers.auth         import router as auth_router
from api_services.routers.chat         import router as chat_router
from config.settings             import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────
    setup_langsmith()                                   # LangSmith tracing
    start_metrics_server(port=settings.prometheus_port) # Prometheus server
    yield
    # ── Shutdown ──────────────────────────────────────────────────────


app = FastAPI(
    title       = "Role-Based RAG API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Mount Prometheus metrics as ASGI endpoint (alternative to separate server)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}