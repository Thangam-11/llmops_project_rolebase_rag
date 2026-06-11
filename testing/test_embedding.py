from src.embedding_layer.embedding_service import EmbeddingService


def main():

    service = EmbeddingService()

    # ── Single text ────────────────────────────────────────────────────
    text = """
    Employees are entitled to
    20 days annual leave.
    """

    vector = service.embed_text(text)

    print("=" * 50)
    print("SINGLE TEXT EMBEDDING")
    print("=" * 50)
    print(f"Text            : {text.strip()}")
    print(f"Vector Dimension: {len(vector)}")
    print(f"First 10 values : {vector[:10]}")

    # ── Batch texts ────────────────────────────────────────────────────
    texts = [
        "What is the employee leave policy?",
        "What is the company revenue in 2024?",
        "What are the marketing campaigns?",
        "What is the system architecture?",
    ]

    vectors = service.embed_texts(texts)

    print("\n" + "=" * 50)
    print("BATCH TEXT EMBEDDING")
    print("=" * 50)
    print(f"Texts count     : {len(texts)}")
    print(f"Vectors count   : {len(vectors)}")
    print(f"Each dimension  : {len(vectors[0])}")

    for i, (t, v) in enumerate(
        zip(texts, vectors),
        start=1,
    ):
        print(f"\n  [{i}] {t}")
        print(f"       First 5 values: {v[:5]}")

    # ── LangChain query embed ──────────────────────────────────────────
    query  = "What is the annual leave entitlement?"
    lc_vec = service.embed_query(query)

    print("\n" + "=" * 50)
    print("LANGCHAIN QUERY EMBEDDING")
    print("=" * 50)
    print(f"Query           : {query}")
    print(f"Vector Dimension: {len(lc_vec)}")
    print(f"First 10 values : {lc_vec[:10]}")

    # ── Dimension check ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("DIMENSION CHECK")
    print("=" * 50)
    print(f"embedding_dimension() : {service.embedding_dimension()}")
    print(f"dimension()           : {service.dimension()}")
    print(f"Actual vector length  : {len(vector)}")

    assert len(vector)    == 768, "❌ Wrong dimension"
    assert len(lc_vec)    == 768, "❌ Wrong LangChain dimension"
    assert len(vectors[0])== 768, "❌ Wrong batch dimension"

    print("\n✅ All dimension checks passed")
    print("=" * 50)


if __name__ == "__main__":
    main()