# testing/check_hr_data.py
from src.embedding_layer.embedding_service import get_embedding_service
from src.vectordb.qdrant_store import QdrantStore
from config.settings import get_settings

settings = get_settings()


def main():

    store = QdrantStore(
        embedding_service=get_embedding_service(),
    )

    docs = store.search(
        query="leave policy",
        department="hr",
        k=5,
    )

    print(f"HR chunks found: {len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print(f"[{i}] score={doc.metadata.get('score', 0):.4f}")
        print(f"     file={doc.metadata.get('filename')}")
        print(f"     content:\n{doc.page_content[:500]}")
        print()


if __name__ == "__main__":
    main()