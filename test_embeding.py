from src.embedding_layer.embedding_service import (
    EmbeddingService,
)

service = EmbeddingService()

text = """
Employees are entitled to
20 days annual leave.
"""

vector = service.embed_text(
    text
)

print(
    f"Vector Dimension: {len(vector)}"
)

print(
    f"First 10 values: {vector[:10]}"
)