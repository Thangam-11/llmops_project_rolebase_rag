FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
COPY pyproject.toml ./
COPY . .
RUN uv pip install --system .
RUN python -m spacy download en_core_web_lg
# Pre-download the embedding model at build time so EmbeddingService's
# cached_property finds it already on disk instead of hitting the
# network on the first /chat/query request.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')"
RUN useradd -m appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1
CMD ["uvicorn", "api_services.main:app", "--host", "0.0.0.0", "--port", "8001"]