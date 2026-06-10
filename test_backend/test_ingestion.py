from pathlib import Path

from src.data_ingestion.data_loader import (
    IngestionService
)

service = IngestionService()

documents = service.ingest_directory(
    Path("data")
)



for doc in documents:

    markdown = (
        doc["document"]
        .export_to_markdown()
    )

    print("=" * 80)

    print(doc["department"])
    print(doc["filename"])

    print(markdown[:300])