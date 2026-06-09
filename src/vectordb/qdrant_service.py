import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

settings = get_settings()


class QdrantService:

    def __init__(self):

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        self.collection_name = settings.collection_name

        logger.info(
            "Qdrant Service Initialized"
        )

    def create_collection(self):

        collections = self.client.get_collections()

        collection_names = [
            c.name
            for c in collections.collections
        ]

        if self.collection_name in collection_names:

            logger.info(
                "Collection already exists"
            )

            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Collection created successfully"
        )

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        department: str,
        filename: str,
    ):

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "department": department,
                        "filename": filename,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["text"],
                        "chunk_type": chunk.get(
                            "chunk_type",
                            "text",
                        ),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(
            f"Inserted {len(points)} chunks"
        )

    def search(
        self,
        query_embedding: list[float],
        department: str,
        limit: int = 5,
    ):

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter={
                "must": [
                    {
                        "key": "department",
                        "match": {
                            "value": department
                        },
                    }
                ]
            },
        )

        return results