from src.vectordb.qdrant_store import QdrantService

qdrant = QdrantService()

info = qdrant.client.get_collection(
    qdrant.collection_name
)

print(info)