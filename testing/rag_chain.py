"""
Diagnose department mismatch and fix it.
Run: python -m testing.check_and_fix
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from config.settings import get_settings

settings = get_settings()

client     = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)
collection = settings.collection_name


def check_departments():

    print("=" * 55)
    print("STEP 1 — What departments exist in Qdrant?")
    print("=" * 55)

    # Scroll all points
    all_points = []
    offset     = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    print(f"Total points : {len(all_points)}")

    # Count by department field
    dept_counts: dict[str, int] = {}

    for point in all_points:
        payload = point.payload or {}

        # LangChain stores metadata nested
        metadata = payload.get("metadata", {})
        dept     = (
            metadata.get("department")
            or payload.get("department")
            or "unknown"
        )
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    print("\nDepartments found:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept:<20} : {count} chunks")

    return dept_counts

# Replace the test_search function with this

def test_search(department: str):

    print(f"\n{'=' * 55}")
    print(f"Test search for dept='{department}'")
    print(f"{'=' * 55}")

    # ✅ Import singleton — no re-loading
    from src.embedding_layer.embedding_service import get_embedding_service
    from src.vectordb.qdrant_store import QdrantStore

    store = QdrantStore(
        embedding_service=get_embedding_service(),
    )

    docs = store.search(
        query="What is the employee leave policy?",
        department=department,
        k=3,
    )

    print(f"Results for '{department}': {len(docs)}")

    for i, doc in enumerate(docs, 1):
        print(
            f"  [{i}] "
            f"file={doc.metadata.get('filename')} | "
            f"dept={doc.metadata.get('department')} | "
            f"score={doc.metadata.get('score', 0):.4f}"
        )

    return docs


def show_sample_payload():

    print(f"\n{'=' * 55}")
    print("STEP 3 — Sample payload structure")
    print(f"{'=' * 55}")

    points, _ = client.scroll(
        collection_name=collection,
        limit=2,
        with_payload=True,
    )

    for i, point in enumerate(points, 1):
        print(f"\nPoint {i} payload:")
        payload = point.payload or {}
        print(f"  Top-level keys : {list(payload.keys())}")

        if "metadata" in payload:
            print(f"  metadata keys  : {list(payload['metadata'].keys())}")
            print(f"  department     : {payload['metadata'].get('department')}")
            print(f"  filename       : {payload['metadata'].get('filename')}")
        else:
            print(f"  department     : {payload.get('department')}")
            print(f"  filename       : {payload.get('filename')}")


def main():

    # Step 1 — see what departments exist
    dept_counts = check_departments()

    # Step 2 — show raw payload structure
    show_sample_payload()

    # Step 3 — test with actual department names found
    print(f"\n{'=' * 55}")
    print("STEP 4 — Test search with found department names")
    print(f"{'=' * 55}")

    for dept in dept_counts.keys():
        test_search(dept)

    # Step 5 — recommendation
    print(f"\n{'=' * 55}")
    print("SUMMARY")
    print(f"{'=' * 55}")

    expected = {"hr", "finance", "marketing", "engineering", "general"}
    found    = set(dept_counts.keys())
    mismatch = found - expected

    if mismatch:
        print(f"⚠️  Mismatched department names: {mismatch}")
        print()
        print("Fix options:")
        print()
        print("  Option A — Re-ingest with correct folder names:")
        print("    1. Rename data/hr_data → data/hr")
        print("    2. python -m src.ingestion_pipeline.ingest_to_qdrant")
        print()
        print("  Option B — Add name mapping in QdrantStore.search():")
        print("    DEPT_MAP = {'hr': 'hr_data', ...}")
    else:
        print("✅ All department names match correctly")


if __name__ == "__main__":
    main()