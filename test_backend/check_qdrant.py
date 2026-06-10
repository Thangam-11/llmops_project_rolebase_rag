from qdrant_client import QdrantClient
from config.settings import get_settings

settings = get_settings()

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)

collection = settings.collection_name

print("=" * 50)
print(f"Collection name : {collection}")

info = client.get_collection(collection)

print(f"Points count    : {info.points_count}")
print(f"Status          : {info.status}")
print(f"Payload schema  : {info.payload_schema}")
print("=" * 50)