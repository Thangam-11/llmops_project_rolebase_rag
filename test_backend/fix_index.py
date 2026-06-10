from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from config.settings import get_settings

settings = get_settings()

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)

collection = settings.collection_name

print(f"Collection : {collection}")

try:
    client.create_payload_index(
        collection_name=collection,
        field_name="department",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("department index created ✓")

except Exception as e:
    print(f"Index creation result: {e}")

# Verify
info = client.get_collection(collection)
print(f"Payload schema now: {info.payload_schema}")