"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.database import engine, Base
from models import model   # register all models
from api_services.routers.auth import router as auth_router
from api_services.routers.chat import router as chat_router
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FinSolve RAG API...")

    # Create all PostgreSQL tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready ✓")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="FinSolve RAG Chatbot API",
    description="Role-Based RAG with JWT + PostgreSQL",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "app":     "FinSolve RAG Chatbot",
        "version": "1.0.0",
    }