# test_qdrant_connection.py
import httpx
from config.settings import get_settings

settings = get_settings()

print("=" * 50)
print(f"Testing connection to: {settings.qdrant_url}")
print("=" * 50)

# Test 1 — HTTP ping
try:
    url  = settings.qdrant_url.rstrip("/") + "/healthz"
    resp = httpx.get(url, timeout=5)
    print(f"✅ HTTP ping OK — status={resp.status_code}")
except Exception as e:
    print(f"❌ HTTP ping failed: {e}")
    print()
    print("Possible fixes:")
    print("  1. Start Docker:  docker run -d -p 6333:6333 qdrant/qdrant")
    print("  2. Check QDRANT_URL in your .env file")
    print("  3. If cloud — verify API key and cluster URL")
    exit(1)

# Test 2 — Qdrant client
try:
    from qdrant_client import QdrantClient

    client      = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    collections = client.get_collections()
    names       = [c.name for c in collections.collections]

    print(f"✅ Qdrant client OK")
    print(f"   Collections: {names if names else '(none yet)'}")

except Exception as e:
    print(f"❌ Qdrant client failed: {e}")
    exit(1)

print()
print("✅ Connection successful — ready to ingest")
print("=" * 50)