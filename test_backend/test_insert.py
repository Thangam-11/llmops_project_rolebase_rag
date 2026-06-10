from src.vectordb.qdrant_service import QdrantService
from src.embedding_layer.embedding_service import EmbeddingService

qdrant = QdrantService()
embedding_service = EmbeddingService()

text = """
Employees are entitled to 20 days annual leave.
"""

vector = embedding_service.embed_text(text)

chunk = {
    "chunk_id": 1,
    "text": text,
    "chunk_type": "text",
}

qdrant.insert_chunks(
    chunks=[chunk],
    embeddings=[vector],
    department="hr",
    filename="employee_handbook.md",
)

print("Inserted Successfully")