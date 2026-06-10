import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,   # ← add this
)

from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
settings = get_settings()


class QdrantService:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.collection_name = settings.collection_name
        logger.info("Qdrant Service Initialized")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self):
        collections = self.client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name in existing:
            logger.info("Collection already exists")
            return

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection created successfully")

        # ── Create payload index on 'department' ──────────────────────
        # Required for filtered search to work
        self._create_payload_index()

    def _create_payload_index(self):
        """
        Create keyword index on 'department' field.
        Must exist before any filtered query_points() call.
        """
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="department",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "Payload index created on 'department' field ✓"
        )

    def ensure_payload_index(self):
        """
        Check if index exists, create if missing.
        Call this on startup to fix existing collections.
        """
        try:
            collection_info = self.client.get_collection(
                self.collection_name
            )
            indexes = collection_info.payload_schema or {}

            if "department" not in indexes:
                logger.info(
                    "department index missing — creating now..."
                )
                self._create_payload_index()
            else:
                logger.info(
                    "department index already exists ✓"
                )
        except Exception as e:
            logger.warning(f"Could not verify index: {e}")

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        department: str,
        filename: str,
    ):
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "department": department,
                        "filename":   filename,
                        "chunk_id":   chunk["chunk_id"],
                        "chunk_text": chunk["text"],
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "headings":   chunk.get("headings", []),
                        "page":       chunk.get("page"),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Inserted {len(points)} chunks")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        department: str,
        limit: int = 5,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="department",
                        match=MatchValue(value=department),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        logger.info(
            f"Found {len(response.points)} results "
            f"for dept='{department}'"
        )

        return response.points