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

# Source must be present before `uv pip install .`, since it installs the
# project itself (not just declared deps) and needs the package files to
# build it. Copying everything first sacrifices some layer caching, but
# the previous order (pyproject.toml only, then install) fails the build
# outright once your build backend tries to package files that aren't there yet.
COPY pyproject.toml ./
COPY . .

RUN uv pip install --system .
RUN python -m spacy download en_core_web_lg

RUN useradd -m appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "api_services.main:app", "--host", "0.0.0.0", "--port", "8001"]