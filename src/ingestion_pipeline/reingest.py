"""
Re-ingest script.
Renames hr_data → hr folder,
deletes the existing Qdrant collection,
then runs the full ingestion pipeline fresh.

Run:
    python reingest.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (# noqa: E402
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from config.settings import get_settings # noqa: E402
from src.ingestion_pipeline.ingest_to_qdrant import IngestionPipeline # noqa: E402

settings = get_settings()


# ── Step 1: Fix folder name ────────────────────────────────────────────────

print("=" * 55)
print("Step 1: Fix HR folder name")
print("=" * 55)

hr_old = Path("data/hr_data")
hr_new = Path("data/hr")

if hr_new.exists():
    print("  ✅ data/hr already exists — skipping")

elif hr_old.exists():
    shutil.copytree(str(hr_old), str(hr_new))
    print("  ✅ Copied data/hr_data → data/hr")

else:
    print("  ⚠️  Neither data/hr nor data/hr_data found")
    print("  Check your data folder structure")


# ── Step 2: Delete old collection ─────────────────────────────────────────

print()
print("=" * 55)
print("Step 2: Delete old collection")
print("=" * 55)

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
    timeout=120,
)

existing = [
    c.name
    for c in client.get_collections().collections
]

if settings.collection_name in existing:
    client.delete_collection(settings.collection_name)
    print(f"  ✅ Deleted '{settings.collection_name}'")
else:
    print(
        f"  Collection '{settings.collection_name}' "
        f"not found — skipping"
    )


# ── Step 3: Recreate collection + index ───────────────────────────────────

print()
print("=" * 55)
print("Step 3: Recreate collection with index")
print("=" * 55)

client.create_collection(
    collection_name=settings.collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
)
print(f"  ✅ Created '{settings.collection_name}'")

client.create_payload_index(
    collection_name=settings.collection_name,
    field_name="metadata.department",
    field_schema=PayloadSchemaType.KEYWORD,
)
print("  ✅ Created metadata.department index")


# ── Step 4: Re-ingest all documents ───────────────────────────────────────

print()
print("=" * 55)
print("Step 4: Re-ingest all documents")
print("=" * 55)

pipeline = IngestionPipeline()
stats = pipeline.run("data")


# ── Step 5: Verify departments ─────────────────────────────────────────────

print()
print("=" * 55)
print("Step 5: Verify inserted departments")
print("=" * 55)

all_points = []
offset     = None

while True:
    points, next_offset = client.scroll(
        collection_name=settings.collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
    )
    all_points.extend(points)
    if next_offset is None:
        break
    offset = next_offset

dept_counts: dict[str, int] = {}

for point in all_points:
    payload  = point.payload or {}
    metadata = payload.get("metadata", {})
    dept     = (
        metadata.get("department")
        or payload.get("department")
        or "unknown"
    )
    dept_counts[dept] = dept_counts.get(dept, 0) + 1

print(f"\n  Total points : {len(all_points)}\n")
print(f"  {'Department':<20} Chunks")
print(f"  {'-' * 30}")

for dept, count in sorted(dept_counts.items()):
    status = "✅" if dept in {
        "engineering", "finance",
        "general", "hr", "marketing",
    } else "⚠️ "
    print(f"  {status} {dept:<18} {count}")

print()

# Check for hr_data mismatch
if "hr_data" in dept_counts:
    print("  ⚠️  hr_data still present — re-ingest did not fix it")
    print("      Manually delete data/hr_data folder and rerun")
elif "hr" in dept_counts:
    print("  ✅ All department names correct")
else:
    print("  ⚠️  hr department missing — check data/hr folder")

print()
print("=" * 55)
print(f"  Files   : {stats.get('files', 0)}")
print(f"  Chunks  : {stats.get('chunks', 0)}")
print("  Re-ingestion complete ✓")
print("=" * 55)