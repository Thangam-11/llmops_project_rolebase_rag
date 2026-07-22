import pytest

from embedding_layer.embedding_service import EmbeddingService


@pytest.fixture(scope="session")
def real_embedding_service():
    """
    Loads the ACTUAL BAAI/bge-base-en-v1.5 model once for the whole
    test session. This is slow (first run downloads/loads weights),
    so scope=session ensures every test in tests/integration/ reuses
    the same loaded model instead of reloading it per test.
    """
    return EmbeddingService(model_name="BAAI/bge-base-en-v1.5", device="cpu")


@pytest.fixture
def qdrant_memory_client():
    """
    Real Qdrant client in in-memory mode — exercises real collection
    creation, real vector indexing, and real similarity search, with
    no external server or Docker dependency.
    """
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    yield client
    # in-memory client has no explicit teardown needed; it's discarded with the fixture


@pytest.fixture
def multi_department_collection(real_embedding_service, qdrant_memory_client):
    """
    A single Qdrant collection with points tagged across multiple
    departments via payload.
    """
    from qdrant_client.models import Distance, VectorParams, PointStruct

    collection_name = "test_multi_dept"
    dim = real_embedding_service.embedding_dimension()

    qdrant_memory_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    documents = [
        ("Q3 revenue was $5 million, up 12% year over year.", "finance", "q3_report.pdf"),
        ("Annual budget allocation increased by 8% for next fiscal year.", "finance", "budget.pdf"),
        ("Employee salaries are reviewed every 6 months.", "hr", "salary_policy.pdf"),
        ("New hires complete onboarding within their first week.", "hr", "onboarding.pdf"),
        ("Our microservices use gRPC for internal communication.", "engineering", "architecture.pdf"),
        ("Code reviews require two approvals before merging.", "engineering", "dev_process.pdf"),
        ("The Q3 marketing campaign increased signups by 30%.", "marketing", "campaign_report.pdf"),
        ("Office hours are 9 AM to 6 PM, Monday through Friday.", "general", "office_policy.pdf"),
    ]

    points = []
    for idx, (text, department, filename) in enumerate(documents):
        vector = real_embedding_service.embed_documents([text])[0]
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={"text": text, "department": department, "filename": filename},
            )
        )

    qdrant_memory_client.upsert(collection_name=collection_name, points=points)

    return collection_name