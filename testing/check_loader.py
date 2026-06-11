from pathlib import Path
from src.data_ingestion.data_loader import DocumentLoader

loader = DocumentLoader()

documents = loader.ingest_directory(Path("data"))

print(f"Total documents loaded: {len(documents)}")

for doc in documents:
    markdown = doc["document"].export_to_markdown()

    print("=" * 80)
    print(f"Department : {doc['department']}")
    print(f"Filename   : {doc['filename']}")
    print(f"File Type  : {doc['file_type']}")
    print(f"Total Chars: {len(markdown)}")
    print(f"Content Preview:")
    print(markdown[:300])