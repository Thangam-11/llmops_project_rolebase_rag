# Role-Based RAG

Department-aware retrieval augmented generation app with:

- FastAPI backend for auth, chat, history, and metrics
- Streamlit frontend for login, registration, chat, and history
- PostgreSQL user/query logging
- Qdrant vector search with department filters
- OpenRouter-compatible LLM calls through LangChain
- PII scrubbing and optional RAGAS evaluation hooks

## Project Layout

```text
api_services/       FastAPI app, routers, and Pydantic schemas
auth/               Password hashing, JWTs, auth dependencies
config/             Environment-backed settings
data/               Department documents for ingestion
models/             SQLAlchemy models and seed users
src/                Ingestion, embeddings, Qdrant, retrieval, RAG chain
app.py              Streamlit frontend
```

## Prerequisites

- Python 3.11+
- PostgreSQL
- Qdrant
- OpenRouter API key

Redis is configured but not required by the current request path.

## Setup

```powershell
copy .env.example .env
uv sync
```

Edit `.env` with your real database URL, Qdrant URL, secret key, and OpenRouter key.

If you are not using `uv`, install the package with pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Database

Create the PostgreSQL database named in `DATABASE_URL`, then create tables and demo users:

```powershell
.\.venv\Scripts\python.exe -m models.seeds
```

Seeded users are listed in [models/seeds.py](models/seeds.py). Example:

- `tony@finsolve.com` / `Tony@Admin1`
- `sam@finsolve.com` / `Finance@123`
- `employee1@finsolve.com` / `Employee@1`

## Ingest Documents

Start Qdrant, then ingest the department files in `data/`:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion_pipeline.ingest_to_qdrant
```

Documents are tagged by department folder, and users can retrieve only from their department plus `general`, except C-level/admin users.

## Run The Backend

```powershell
.\.venv\Scripts\uvicorn.exe api_services.main:app --host 0.0.0.0 --port 8001 --reload
```

Useful endpoints:

- `GET /health`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /chat/query`
- `GET /chat/history`
- `GET /metrics`

## Run The Frontend

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The frontend defaults to `http://localhost:8001`. Override it with:

```powershell
$env:ROLE_RAG_API_BASE="http://localhost:8001"
.\.venv\Scripts\streamlit.exe run app.py
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m compileall api_services auth config models src app.py
```

For an end-to-end check:

1. Start PostgreSQL and Qdrant.
2. Run `models.seeds`.
3. Run ingestion.
4. Start the backend on port `8001`.
5. Start Streamlit.
6. Log in as a seeded user and ask a department-specific question.

## Notes

- New frontend registrations create `viewer` users, matching the backend role enum.
- Department routing is enforced by `RetrieverService` and Qdrant payload filters.
- `.env` is intentionally ignored by git; use `.env.example` as the shared template.
