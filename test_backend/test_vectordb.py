# test_qdrant.py

from src.vectordb.qdrant_store import (
    QdrantService
)

qdrant = QdrantService()

qdrant.create_collection()

print("Collection Created Successfully")