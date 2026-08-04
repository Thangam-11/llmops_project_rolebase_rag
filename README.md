# Role-Based RAG

A department-aware **Retrieval Augmented Generation (RAG)** application that lets employees ask natural-language questions and get answers grounded in company documents — while enforcing role- and department-based access control on what each user can retrieve.

![Architecture](docs/images/architecture.png)

The system combines a **FastAPI** backend, a **Streamlit** frontend, **PostgreSQL** for user/query logging, **Qdrant** for vector search (with **BAAI/bge-base-en-v1.5** embeddings), and **LangChain** for orchestrating LLM calls to **Gemini**, **OpenAI GPT-4o**, **LLaMA 3**, or any OpenRouter-compatible model. Safety and quality are enforced with **NeMo Guardrails**, **Presidio** PII scrubbing, and **RAGAS** evaluation, with **LangSmith**, **Prometheus**, and **Grafana** for observability. The app ships with a full **GitHub Actions** CI/CD pipeline (Ruff, mypy, pytest, Docker smoke test) and deploys to **Azure Container Apps**, with secrets managed in **Azure Key Vault**.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Database](#database)
- [Ingest Documents](#ingest-documents)
- [Run the Backend](#run-the-backend)
- [Run the Frontend](#run-the-frontend)
- [Running Everything with Docker Compose](#running-everything-with-docker-compose)
- [Role-Based Access Control](#role-based-access-control)
- [API Reference](#api-reference)
- [PII Scrubbing, Guardrails & Evaluation](#pii-scrubbing-guardrails--evaluation)
- [Observability](#observability)
- [Testing & Code Quality](#testing--code-quality)
- [CI/CD & Deployment](#cicd--deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- 🔐 **JWT-based authentication** — register, login, and session management via FastAPI
- 🏢 **Department-aware retrieval** — users only retrieve documents from their own department (HR, Finance, Engineering, Marketing) plus `general`, unless they hold a C-Level/admin role
- 🔎 **Vector search with Qdrant** — semantic search over ingested department documents using **BAAI/bge-base-en-v1.5** embeddings, filtered by department metadata
- 📄 **Document ingestion pipeline** — PDF/DOCX sources are chunked, embedded, and upserted into Qdrant
- 🤖 **LLM orchestration with LangChain** — RAG chain supports **Gemini**, **OpenAI GPT-4o**, and **LLaMA 3** (8B/70B), or any OpenRouter-compatible model
- 🗂️ **Query & chat history logging** — all interactions persisted to PostgreSQL for auditing and analytics
- ⚡ **Response caching with Redis** — repeated/similar queries can be served from cache to reduce latency and LLM cost
- 🧼 **PII scrubbing with Presidio** — Microsoft Presidio detects and redacts sensitive information (names, emails, financial identifiers, etc.) before it reaches the LLM or logs
- 🛡️ **NeMo Guardrails** — programmable guardrails wrap the RAG chain to block prompt injection, detect PII, filter toxic content, and validate responses before they reach the user
- 📊 **RAGAS evaluation** — integrated evaluation of retrieval and generation quality (faithfulness, answer relevancy, context precision/recall)
- 🔭 **LangSmith tracing** — end-to-end tracing and observability of LLM calls and chains
- 📈 **Prometheus & Grafana** — operational metrics collection and dashboards, alongside a lightweight `/metrics` API endpoint
- 🖥️ **Streamlit frontend** — simple UI for login, registration, chat, and history browsing
- 🐳 **Docker Compose** — one-command local orchestration of backend, frontend, Postgres, Qdrant, and Redis
- ✅ **CI pipeline** — GitHub Actions run linting (Ruff), type checking (mypy), unit tests (pytest), and a Docker smoke test on every change
- ☁️ **Azure deployment** — containerized app is built and deployed to an Azure Container instance/app via GitHub Actions, with **Azure Key Vault** for secrets management

---

## Architecture Overview

> **LLM options:** Gemini · OpenAI GPT-4o · LLaMA 3 (8B / 70B)
> **Embedding model:** BAAI/bge-base-en-v1.5 (768 dimensions)

The application follows a 12-step request flow, backed by an offline document ingestion pipeline and a supporting infrastructure/observability layer.

### 1. User Interaction

Users from different departments access the application via the **Streamlit frontend**: HR, Finance, Engineering, Marketing, and C-Level users all share the same chat UI.

### 2. Authentication & Authorization

The user logs in; a **JWT** is generated and the user's role and department are identified. Users, roles, departments, and permissions are stored in **PostgreSQL**.

### 3. Role-Based Access Control

The system checks the user's role and grants access only to permitted departments and documents, via a role-to-department mapping:

| Role       | Accessible Documents |
|------------|------------------------|
| HR         | HR Docs                 |
| Finance    | Finance Docs            |
| Engineering| Engineering Docs        |
| Marketing  | Marketing Docs          |
| C-Level    | All Departments          |

### 4. Query Processing & Guardrails

The user's query passes through multiple guardrails to ensure safety and compliance before retrieval:

- Prompt injection detection
- PII detection (**Presidio**)
- Toxicity / hate speech filtering
- Off-topic check

### 5. Query Embedding

The validated query is converted into a vector representation using **BAAI/bge-base-en-v1.5** (768 dimensions), e.g. `[0.12, -0.34, 0.56, ..., 0.02]`.

### 6. Vector Search (Qdrant)

The query vector is searched against **Qdrant**, scoped to the user's authorized departments (HR, Finance, Engineering, Marketing, General), using per-department Qdrant collections.

### 7. Context Retrieval

The top-K most relevant document chunks are retrieved along with metadata and similarity scores, e.g.:

| Chunk            | Score |
|-------------------|--------|
| Document Chunk 1 | 0.92    |
| Document Chunk 2 | 0.88    |
| Document Chunk 3 | 0.85    |
| ...               | ...     |
| Document Chunk K | 0.76    |

### 8. Prompt Construction

The retrieved context, the user's query, and system instructions are combined into a final prompt using a template that includes: system instructions, user query, retrieved context (top-K), and role/department info.

### 9. LLM Generation

The prompt is sent to the selected LLM to generate the answer:

- **Gemini**
- **OpenAI GPT-4o**
- **LLaMA 3** (8B / 70B)

### 10. Response Validation

The generated response is validated for quality and safety:

- Hallucination check
- Answer relevance
- Source attribution check
- PII / sensitive information check
- Format / policy compliance

### 11. Response to User

The final, validated answer is returned to the user in Streamlit along with its cited sources (e.g. `Document.pdf`, `Policy.docx`, `Handbook.pdf`).

### 12. Logging & Analytics

Query, response, sources, latency, and feedback are logged for monitoring and improvement:

- **PostgreSQL** — query logs
- **Redis** — response cache
- Analytics & reports

---

### Document Ingestion Pipeline (Offline)

Documents are ingested and indexed independently of the live request path:

```text
1. Data Sources          PDF, DOCX, PPT, TXT, websites, etc.
2. Document Parsing      Docling / document converter
3. Chunking              Recursive / semantic chunking
4. Embedding              BAAI/bge-base-en-v1.5 (768 dims)
5. Store in Vector DB    Qdrant collections (per department)
```

---

### Supporting Infrastructure

| Component          | Role                                    |
|----------------------|--------------------------------------------|
| **Docker Containers**| Containerized deployment of all services   |
| **FastAPI**          | Backend API                                 |
| **Streamlit**        | Frontend                                    |
| **PostgreSQL**       | Database (users, roles, query logs)         |
| **Redis**            | Response cache                              |
| **Qdrant**           | Vector database                             |

### Observability & Monitoring

| Component      | Role                                  |
|------------------|------------------------------------------|
| **Prometheus**  | Metrics collection                         |
| **Grafana**      | Dashboards & alerts                        |
| **LangSmith**    | LLM tracing & evaluation                   |
| Logging          | Application logs                           |

### External Services

| Component            | Role                              |
|------------------------|--------------------------------------|
| **Azure / AWS**       | Cloud infrastructure                  |
| **Azure Key Vault**   | Secrets management                     |

---

## Project Layout

```text
api_services/       FastAPI app, routers, and Pydantic schemas
auth/                Password hashing, JWTs, auth dependencies
config/              Environment-backed settings
data/                Department documents for ingestion
models/              SQLAlchemy models and seed users
src/                 Ingestion, embeddings, Qdrant, retrieval, RAG chain
app.py               Streamlit frontend
```

| Directory / File     | Purpose                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `api_services/`      | FastAPI application entrypoint, routers (`auth`, `chat`, `history`, `metrics`), and Pydantic request/response schemas |
| `auth/`               | Password hashing utilities, JWT creation/validation, FastAPI auth dependencies |
| `config/`             | Centralized, environment-variable-backed settings (via `.env`)          |
| `data/`               | Source documents organized by department folder, used for ingestion    |
| `models/`             | SQLAlchemy ORM models and `seeds.py` for demo/test users                |
| `src/`                | Ingestion pipeline, embedding generation, Qdrant client, retrieval logic, and the RAG chain construction |
| `app.py`              | Streamlit UI: login, registration, chat interface, and history viewer   |

---

## Prerequisites

- **Python 3.11+**
- **PostgreSQL** (running instance + a database created for this app)
- **Qdrant** (running instance, local via Docker or hosted)
- **Redis** (used for response caching)
- **API key(s) for your chosen LLM provider(s)** — Gemini, OpenAI, and/or OpenRouter
- **Docker & Docker Compose** (recommended for local orchestration of all services)
- *(Optional)* **LangSmith account/API key** for tracing
- *(Optional)* **Prometheus & Grafana** for metrics dashboards

---

## Setup

Clone the repository, then copy the environment template and install dependencies.

```powershell
copy .env.example .env
uv sync
```

If you're not using [`uv`](https://github.com/astral-sh/uv), install dependencies with `pip` instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> On macOS/Linux, replace the PowerShell activation step with:
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> ```

---

## Environment Variables

Edit `.env` with your real values before running the app:

| Variable            | Description                                              | Example                                              |
|---------------------|------------------------------------------------------------|-------------------------------------------------------|
| `DATABASE_URL`      | PostgreSQL connection string                                | `postgresql://user:password@localhost:5432/rag_db`   |
| `QDRANT_URL`        | Qdrant instance URL                                         | `http://localhost:6333`                                |
| `SECRET_KEY`        | Secret used to sign JWTs                                    | `change-me-to-a-long-random-string`                   |
| `OPENROUTER_API_KEY`| API key for OpenRouter (or compatible provider)              | `sk-or-...`                                            |
| `GEMINI_API_KEY`    | API key for Google Gemini                                    | `AIza...`                                               |
| `OPENAI_API_KEY`    | API key for OpenAI                                            | `sk-...`                                                |
| `LLAMA_API_URL`     | Endpoint for the LLaMA 3 (8B/70B) inference server             | `http://localhost:11434`                                |
| `EMBEDDING_MODEL`   | Embedding model used for vectorization                       | `BAAI/bge-base-en-v1.5`                                 |
| `REDIS_URL`         | Redis connection string, used for response caching            | `redis://localhost:6379/0`                              |
| `LANGSMITH_API_KEY` | (Optional) API key for LangSmith tracing                     | `ls__...`                                               |
| `LANGSMITH_PROJECT` | (Optional) LangSmith project name                             | `role-based-rag`                                        |
| `AZURE_KEY_VAULT_URL` | (Optional) Azure Key Vault URL for centralized secrets       | `https://<vault-name>.vault.azure.net/`                 |

> Adjust variable names to match those actually referenced in `config/`; update this table if your `.env.example` differs.

---

## Database

Create the PostgreSQL database named in `DATABASE_URL`, then create tables and seed demo users:

```powershell
.\.venv\Scripts\python.exe -m models.seeds
```

This will:

- Create all tables defined by the SQLAlchemy models
- Insert a set of demo users across departments and roles

Seeded users are listed in [`models/seeds.py`](models/seeds.py). Example accounts:

| Email                     | Password       | Role / Department        |
|----------------------------|----------------|----------------------------|
| `tony@finsolve.com`       | `Tony@Admin1`  | Admin / C-level (full access) |
| `sam@finsolve.com`        | `Finance@123`  | Finance department         |
| `employee1@finsolve.com`  | `Employee@1`   | General employee           |

> ⚠️ These are demo credentials for local development only — rotate or remove them before deploying to production.

---

## Ingest Documents

Start Qdrant, then ingest the department files in `data/`:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion_pipeline.ingest_to_qdrant
```

The ingestion pipeline:

1. Reads **PDF** and **DOCX** source documents from department-named subfolders under `data/`
2. Splits each document into chunks
3. Generates embeddings for each chunk using **BAAI/bge-base-en-v1.5**
4. Upserts the vectors into the configured **Qdrant** collection, tagged with the source department as metadata

Retrieval later filters on this department tag: users can retrieve only from their own department plus `general`, **except** C-level/admin users, who can retrieve across all departments.

---

## Run the Backend

```powershell
.\.venv\Scripts\uvicorn.exe api_services.main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://localhost:8001`.

### Useful Endpoints

| Method | Path            | Description                                  |
|--------|-----------------|-----------------------------------------------|
| GET    | `/health`       | Health check                                    |
| POST   | `/auth/register`| Register a new user                             |
| POST   | `/auth/login`   | Authenticate and receive a JWT                 |
| GET    | `/auth/me`      | Get the current authenticated user's profile   |
| POST   | `/chat/query`   | Submit a RAG query and receive an answer       |
| GET    | `/chat/history` | Retrieve the current user's chat history       |
| GET    | `/metrics`      | Retrieve basic usage/operational metrics       |

Interactive API docs (Swagger UI) are available at `http://localhost:8001/docs` once the backend is running, and ReDoc at `http://localhost:8001/redoc`.

---

## Run the Frontend

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The Streamlit app will open at `http://localhost:8501` by default.

The frontend defaults to `http://localhost:8001` for the backend API. Override it with:

```powershell
$env:BACKEND_URL="http://localhost:8001"
.\.venv\Scripts\streamlit.exe run app.py
```

> On macOS/Linux:
> ```bash
> export BACKEND_URL="http://localhost:8001"
> streamlit run app.py
> ```

---

## Role-Based Access Control

- Every user belongs to a **department** (e.g., Finance, Engineering, HR) recorded at registration/seed time.
- Retrieval queries are automatically filtered in Qdrant so a user only sees results tagged with:
  - their own department, and
  - the shared `general` department
- **C-level and admin users** bypass this filter and can retrieve across **all** departments.
- Access checks are enforced server-side in the backend — the frontend has no independent access logic.

---

## PII Scrubbing, Guardrails & Evaluation

### PII Scrubbing — Presidio

Sensitive information (names, emails, phone numbers, financial identifiers, etc.) is detected and redacted using **Microsoft Presidio**:

- User queries and/or retrieved context can be scrubbed **before** being sent to the LLM.
- Scrubbing is also applied before persisting queries/responses to PostgreSQL logs, reducing the risk of storing sensitive data at rest.
- Detection recognizers and redaction behavior are configurable in `src/`.

### Guardrails — NeMo Guardrails

**NeMo Guardrails** wraps the RAG chain to enforce safe, on-policy behavior:

- Constrains the assistant to approved topics relevant to the RAG use case.
- Blocks jailbreak attempts and prompt-injection patterns in user input.
- Can validate or rewrite outputs before they're returned to the user.
- Guardrail configuration (rails, flows, prompts) lives alongside the RAG chain logic in `src/`.

### Evaluation — RAGAS

**RAGAS** is integrated to evaluate the RAG pipeline's quality, covering metrics such as:

- **Faithfulness** — whether the answer is grounded in the retrieved context
- **Answer relevancy** — whether the answer actually addresses the question
- **Context precision / recall** — whether retrieval surfaced the right chunks

Evaluation can be run offline against a test set of queries to track regressions as the ingestion pipeline, embeddings, or prompts change.

---

## Running Everything with Docker Compose

For local development, `docker-compose.yml` can orchestrate the full stack — backend, frontend, PostgreSQL, Qdrant, and Redis — in one command:

```powershell
docker compose up --build
```

This brings up:

| Service     | Default Port |
|-------------|----------------|
| FastAPI backend | `8001` |
| Streamlit frontend | `8501` |
| PostgreSQL | `5432` |
| Qdrant | `6333` |
| Redis | `6379` |

Stop and remove containers with:

```powershell
docker compose down
```

> Adjust service names/ports to match your actual `docker-compose.yml`.

---

## Observability

- **LangSmith** — traces every LLM call and chain execution (query processing, guardrail checks, prompt construction, generation) for debugging and quality review. Enable by setting `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`.
- **Prometheus** — scrapes operational metrics exposed by the backend (e.g., request counts, latency, error rates) in addition to the `/metrics` API endpoint.
- **Grafana** — visualizes Prometheus metrics in dashboards for monitoring system health and usage trends.

---

## Testing & Code Quality

The project enforces code quality and correctness locally and in CI using:

| Tool     | Purpose                                  | Typical command                     |
|----------|--------------------------------------------|----------------------------------------|
| **Ruff** | Linting (and formatting)                    | `ruff check .`                          |
| **mypy** | Static type checking                        | `mypy .`                                |
| **pytest** | Unit / integration tests                  | `pytest`                                |

Run all checks locally before pushing:

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe .
.\.venv\Scripts\pytest.exe
```

> On macOS/Linux: `ruff check .`, `mypy .`, `pytest` (with your virtual environment activated), or prefix with `uv run` if using `uv`.

### Docker Smoke Test

A lightweight smoke test builds the Docker image and verifies the container starts and responds on `/health`:

```powershell
docker build -t role-based-rag:local .
docker run -d --name role-based-rag-smoke -p 8001:8001 --env-file .env role-based-rag:local
curl http://localhost:8001/health
docker rm -f role-based-rag-smoke
```

This same sequence runs automatically in CI on every pull request.

---

## CI/CD & Deployment

CI/CD is implemented with **GitHub Actions**. On every push/pull request, the pipeline:

1. **Lint** — runs `ruff check .` to catch style and correctness issues
2. **Type check** — runs `mypy .` to verify type correctness
3. **Test** — runs `pytest` for unit and integration tests
4. **Docker smoke test** — builds the Docker image, starts the container, and verifies `/health` responds successfully
5. **Deploy (on merge to main)** — builds and pushes the container image, then deploys it to an **Azure Container** (e.g., Azure Container Apps / Azure Container Instances)

A typical workflow structure (`.github/workflows/ci.yml`):

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-type-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: mypy .
      - run: pytest

  docker-smoke-test:
    needs: lint-type-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t role-based-rag:ci .
      - run: docker run -d --name smoke -p 8001:8001 role-based-rag:ci
      - run: sleep 5 && curl --fail http://localhost:8001/health
      - run: docker rm -f smoke

  deploy-azure:
    if: github.ref == 'refs/heads/main'
    needs: docker-smoke-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      - run: |
          az acr build --registry <your-acr-name> --image role-based-rag:${{ github.sha }} .
      - uses: azure/container-apps-deploy-action@v1
        with:
          containerAppName: role-based-rag
          resourceGroup: <your-resource-group>
          imageToDeploy: <your-acr-name>.azurecr.io/role-based-rag:${{ github.sha }}
```

> Replace `<your-acr-name>`, `<your-resource-group>`, and secret names with the values used in your actual Azure setup. Store credentials (`AZURE_CREDENTIALS`, registry login, etc.) as GitHub Actions secrets — never commit them.

---

## Troubleshooting

| Symptom                                   | Likely Cause / Fix                                                        |
|--------------------------------------------|------------------------------------------------------------------------------|
| Backend fails to start with a DB error    | Verify `DATABASE_URL` and that PostgreSQL is running and reachable          |
| `/chat/query` returns empty results        | Confirm documents were ingested (`src.ingestion_pipeline.ingest_to_qdrant`) and Qdrant is running |
| 401 Unauthorized on protected endpoints   | Ensure you're sending a valid JWT from `/auth/login` in the `Authorization` header |
| Frontend can't reach backend               | Check `BACKEND_URL` matches the host/port the backend is running on         |
| LLM calls fail                             | Verify `OPENROUTER_API_KEY` is set correctly in `.env`                       |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes with clear, focused commits.
3. Ensure the backend and frontend both run locally without errors.
4. Open a pull request describing the change and its motivation.

---

## License

Specify your project's license here (e.g., MIT, Apache 2.0). If none is chosen yet, add a `LICENSE` file to the repository root.
