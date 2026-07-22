# save as check_structure.py and run it
from pathlib import Path

required_files = [
    "src/__init__.py",
    "src/ingestion_pipeline/__init__.py",
    "src/ingestion_pipeline/ingest_to_qdrant.py",
    "src/vectordb/__init__.py",
    "src/vectordb/qdrant_store.py",
    "src/data_ingestion/__init__.py",
    "src/data_ingestion/data_loader.py",
    "src/data_ingestion/chunker_service.py",
    "src/embedding_layer/__init__.py",
    "src/embedding_layer/embedding_service.py",
    "config/settings.py",
    "utils/logger_exceptions.py",
]

print("=" * 50)
print("FILE STRUCTURE CHECK")
print("=" * 50)

all_ok = True

for f in required_files:
    path   = Path(f)
    exists = path.exists()
    size   = path.stat().st_size if exists else 0
    status = "✅" if exists else "❌ MISSING"

    print(f"{status}  {f}  ({size} bytes)")

    if not exists:
        all_ok = False

print("=" * 50)

# Check __init__.py files for bad imports
init_files = [
    "src/__init__.py",
    "src/ingestion_pipeline/__init__.py",
    "src/vectordb/__init__.py",
    "src/data_ingestion/__init__.py",
    "src/embedding_layer/__init__.py",
]

print("\nINIT FILE CONTENT CHECK")
print("=" * 50)

for f in init_files:
    path = Path(f)
    if not path.exists():
        print(f"❌ MISSING : {f}")
        continue

    content = path.read_text().strip()

    if content:
        print(f"⚠️  NOT EMPTY : {f}")
        print(f"   Content: {content[:100]}")
        print("   → Clear this file completely")
    else:
        print(f"✅ Empty    : {f}")

print("=" * 50)