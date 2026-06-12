"""
Re-ingest script.
Deletes the existing Qdrant collection (bad metadata)
then runs the full ingestion pipeline fresh.

Run:
    python reingest.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from config.settings import get_settings
from src.ingestion_pipeline.ingest_to_qdrant import IngestionPipeline

settings = get_settings()

# Step 1: delete old collection
print("=" * 50)
print("Step 1: Deleting old collection...")

client = QdrantClient(
    url     = settings.qdrant_url,
    api_key = settings.qdrant_api_key or None,
)

existing = [c.name for c in client.get_collections().collections]

if settings.collection_name in existing:
    client.delete_collection(settings.collection_name)
    print(f"  Deleted '{settings.collection_name}' ✓")
else:
    print(f"  Collection '{settings.collection_name}' not found — skipping")

# Step 2: re-ingest with correct metadata
print()
print("=" * 50)
print("Step 2: Re-ingesting with correct metadata...")

pipeline = IngestionPipeline()
stats    = pipeline.run("data")

print()
print("=" * 50)
print(f"  Files   : {stats['files']}")
print(f"  Chunks  : {stats['chunks']}")
print("  Re-ingestion complete ✓")