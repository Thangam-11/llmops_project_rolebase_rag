# tests/integration/test_qdrant.py
import pytest
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

pytestmark = pytest.mark.integration


def test_create_collection_with_correct_vector_size(real_embedding_service, qdrant_memory_client):
    dim = real_embedding_service.embedding_dimension()
    qdrant_memory_client.create_collection(
        collection_name="test_basic",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    info = qdrant_memory_client.get_collection("test_basic")
    assert info.config.params.vectors.size == dim


def test_upsert_then_retrieve_by_id_returns_exact_payload(real_embedding_service, qdrant_memory_client):
    dim = real_embedding_service.embedding_dimension()
    qdrant_memory_client.create_collection(
        collection_name="test_retrieve",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    vector = real_embedding_service.embed_documents(["some content"])[0]
    qdrant_memory_client.upsert(
        collection_name="test_retrieve",
        points=[PointStruct(id=1, vector=vector, payload={"filename": "doc.pdf", "department": "finance"})],
    )
    results = qdrant_memory_client.retrieve(collection_name="test_retrieve", ids=[1])
    assert len(results) == 1
    assert results[0].payload["filename"] == "doc.pdf"
    assert results[0].payload["department"] == "finance"


def test_upsert_same_id_replaces_not_duplicates(real_embedding_service, qdrant_memory_client):
    dim = real_embedding_service.embedding_dimension()
    qdrant_memory_client.create_collection(
        collection_name="test_replace",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    v1 = real_embedding_service.embed_documents(["original content"])[0]
    v2 = real_embedding_service.embed_documents(["updated content"])[0]
    qdrant_memory_client.upsert(
        collection_name="test_replace",
        points=[PointStruct(id=1, vector=v1, payload={"filename": "v1.pdf"})],
    )
    qdrant_memory_client.upsert(
        collection_name="test_replace",
        points=[PointStruct(id=1, vector=v2, payload={"filename": "v2.pdf"})],
    )
    count = qdrant_memory_client.count(collection_name="test_replace").count
    result = qdrant_memory_client.retrieve(collection_name="test_replace", ids=[1])
    assert count == 1
    assert result[0].payload["filename"] == "v2.pdf"


def test_search_on_empty_collection_returns_empty_list(real_embedding_service, qdrant_memory_client):
    dim = real_embedding_service.embedding_dimension()
    qdrant_memory_client.create_collection(
        collection_name="test_empty",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    query_vector = real_embedding_service.embed_query("anything")
    results = qdrant_memory_client.query_points(
        collection_name="test_empty", query=query_vector, limit=5,
    ).points
    assert results == []


def test_finance_filter_never_returns_hr_documents(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("What is the revenue and salary information?")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchValue(value="finance"))]
        ),
        limit=10,
    ).points
    assert len(results) > 0
    assert all(r.payload["department"] == "finance" for r in results)


def test_hr_filter_never_returns_finance_documents(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("Tell me about compensation and hiring.")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchValue(value="hr"))]
        ),
        limit=10,
    ).points
    assert len(results) > 0
    assert all(r.payload["department"] == "hr" for r in results)


@pytest.mark.parametrize("department", ["finance", "hr", "engineering", "marketing", "general"])
def test_single_department_filter_only_returns_that_department(
    real_embedding_service, qdrant_memory_client, multi_department_collection, department
):
    query_vector = real_embedding_service.embed_query("company information")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchValue(value=department))]
        ),
        limit=10,
    ).points
    assert all(r.payload["department"] == department for r in results)


def test_multi_department_filter_for_c_level_access(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("Give me a company overview")
    allowed_departments = ["engineering", "hr", "finance", "marketing", "general"]
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchAny(any=allowed_departments))]
        ),
        limit=10,
    ).points
    returned_departments = {r.payload["department"] for r in results}
    assert returned_departments.issubset(set(allowed_departments))
    assert len(returned_departments) > 1


def test_engineering_department_filter_excludes_all_others(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("company processes and policies")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchAny(any=["engineering", "general"]))]
        ),
        limit=10,
    ).points
    returned_departments = {r.payload["department"] for r in results}
    assert returned_departments.issubset({"engineering", "general"})
    assert "finance" not in returned_departments
    assert "hr" not in returned_departments
    assert "marketing" not in returned_departments


def test_filter_with_no_matching_department_returns_empty(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("anything at all")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchValue(value="nonexistent_dept"))]
        ),
        limit=10,
    ).points
    assert results == []


def test_no_filter_returns_results_across_all_departments(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("company information")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        limit=10,
    ).points
    returned_departments = {r.payload["department"] for r in results}
    assert len(returned_departments) > 1


def test_score_threshold_filters_low_relevance_results(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("Q3 financial revenue numbers")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        score_threshold=0.5,
        limit=10,
    ).points
    assert all(r.score >= 0.5 for r in results)


def test_top_k_limit_is_respected(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("company information")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        limit=3,
    ).points
    assert len(results) <= 3


def test_results_are_ordered_by_descending_score(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("Q3 revenue numbers")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        limit=10,
    ).points
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_department_filter_combined_with_score_threshold(
    real_embedding_service, qdrant_memory_client, multi_department_collection
):
    query_vector = real_embedding_service.embed_query("hiring and onboarding process")
    results = qdrant_memory_client.query_points(
        collection_name=multi_department_collection,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="department", match=MatchValue(value="hr"))]
        ),
        score_threshold=0.3,
        limit=10,
    ).points
    assert all(r.payload["department"] == "hr" for r in results)
    assert all(r.score >= 0.3 for r in results)