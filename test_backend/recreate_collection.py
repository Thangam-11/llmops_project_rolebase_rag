from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
)
from config.settings import get_settings

settings = get_settings()

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)

collection = settings.collection_name

print("=" * 50)

# Step 1 — delete old collection
try:
    client.delete_collection(collection)
    print(f"Deleted : {collection}")
except Exception as e:
    print(f"Delete skipped : {e}")

# Step 2 — create fresh collection
client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
)
print(f"Created : {collection}")

# Step 3 — create department index
client.create_payload_index(
    collection_name=collection,
    field_name="department",
    field_schema=PayloadSchemaType.KEYWORD,
)
print("department index created ✓")

# Step 4 — verify
info = client.get_collection(collection)
print(f"Points count   : {info.points_count}")
print(f"Payload schema : {info.payload_schema}")
print("=" * 50)
print("Collection ready — now run ingest_all.py")