# test_qdrant.py

from src.vectordb.qdrant_service import (
    QdrantService
)

qdrant = QdrantService()

qdrant.create_collection()

print("Collection Created Successfully")