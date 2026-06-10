from src.retrieval.retriever_service import (
    RetrieverService,
)

retriever = RetrieverService()

question = (
    "What is the employee leave policy?"
)

results = retriever.retrieve(
    question=question,
    department="hr",
    limit=5,
)

print("\nRESULTS\n")

for idx, result in enumerate(
    results,
    start=1,
):

    print(
        f"\nResult {idx}"
    )

    print(
        f"Score: {result['score']}"
    )

    print(
        f"Department: {result['department']}"
    )

    print(
        f"File: {result['filename']}"
    )

    print(
        f"Text:\n"
        f"{result['chunk_text'][:300]}"
    )