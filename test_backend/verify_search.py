from src.vectordb.qdrant_store import QdrantService
from src.embedding_layer.embedding_service import EmbeddingService


def main():

    embedder = EmbeddingService()
    qdrant   = QdrantService()

    question   = "What is the employee leave policy?"
    department = "hr"

    print("=" * 50)
    print(f"Question   : {question}")
    print(f"Department : {department}")
    print("=" * 50)

    # Embed
    vector = embedder.embed_text(question)
    print(f"Vector dim : {len(vector)}")

    # Search
    points = qdrant.search(
        query_embedding=vector,
        department=department,
        limit=3,
    )

    print(f"Results    : {len(points)}\n")

    for i, point in enumerate(points, start=1):
        print(f"Result {i}")
        print(f"  Score      : {point.score:.4f}")
        print(f"  Department : {point.payload.get('department')}")
        print(f"  File       : {point.payload.get('filename')}")
        print(f"  Text       : {point.payload.get('chunk_text', '')[:200]}")
        print()

    print("=" * 50)
    print("Search test complete ✓")


if __name__ == "__main__":
    main()