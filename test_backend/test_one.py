
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from config.settings import get_settings

settings = get_settings()

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
)

collection = settings.collection_name
print('Collection:', collection)

# Force create keyword index on department
client.create_payload_index(
    collection_name=collection,
    field_name='department',
    field_schema=PayloadSchemaType.KEYWORD,
)

print('Index created successfully')

# Verify it was created
info = client.get_collection(collection)
print('Payload schema now:', info.payload_schema)
