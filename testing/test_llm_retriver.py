from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding_layer.embedding_service import EmbeddingService
from src.vectordb.qdrant_store        import QdrantStore
from src.retrieval.retriever_service          import RetrieverService
from src.llm_layer.llm_connecter            import LLMConnector
from config.settings                  import get_settings

settings = get_settings()

# ── 1. Setup ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading embedding model...")
embedder = EmbeddingService()

print("=" * 60)
print("STEP 2: Connecting to Qdrant...")
store = QdrantStore(
    embedding_service = embedder,
    url               = settings.qdrant_url,
    api_key           = settings.qdrant_api_key,
    collection_name   = settings.collection_name,
)

info = store.collection_info()
print(f"  Collection : {info['name']}")
print(f"  Vectors    : {info['vector_count']}")
print(f"  Status     : {info['status']}")

retriever = RetrieverService(store=store, default_k=3)

# ── 2. Test retrieval per department ─────────────────────────────────────────

TEST_QUERIES = [
    ("What is the Q4 revenue?",              "finance"),
    ("What is the leave policy?",            "hr"),
    ("What were the campaign results?",      "marketing"),
    ("What is the system architecture?",     "engineering"),
    ("What are the company benefits?",       "general"),
    ("Give me a summary across all teams.",  "c_level"),
]

print("\n" + "=" * 60)
print("STEP 3: Testing retrieval...")

for question, department in TEST_QUERIES:

    print("\n" + "─" * 60)
    print(f"  Question   : {question}")
    print(f"  Department : {department}")

    docs = retriever.retrieve(question=question, department=department)

    print(f"  Chunks     : {len(docs)}")

    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        print(f"\n    [{i}] {m.get('filename', 'unknown')}"
              f" | dept={m.get('department', '?')}"
              f" | score={m.get('score', 0.0)}"
              f" | page={m.get('page')}")
        print(f"        {doc.page_content[:150].strip()}...")

# ── 3. Test context formatting ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Testing format_context()...")

docs = retriever.retrieve("What is the Q4 revenue?", department="finance")
context = RetrieverService.format_context(docs)

print(context[:600])

# ── 4. Test LLM connector ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Testing LLM health check...")

llm_connector = LLMConnector()
ok = llm_connector.health_check()
print(f"  LLM reachable : {ok}")

print("\n" + "=" * 60)
print("STEP 6: Testing LLM invoke...")

llm = llm_connector.get_llm()
response = llm.invoke("Say hello in one sentence.")
print(f"  Model    : {settings.llm_model}")
print(f"  Response : {response.content}")

# ── 5. Test full retrieval + LLM together ────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Testing retrieval + LLM together...")

question   = "What is the Q4 revenue?"
department = "finance"

docs    = retriever.retrieve(question=question, department=department)
context = RetrieverService.format_context(docs)

prompt = f"""Answer only from the context below.

Context:
{context}

Question:
{question}

Answer:"""

response = llm.invoke(prompt)

print(f"  Question : {question}")
print(f"  Answer   : {response.content}")
print(f"  Sources  :")
for doc in docs:
    print(f"    - {doc.metadata.get('filename')} | page={doc.metadata.get('page')}")

print("\n" + "=" * 60)
print("All tests done ✓")