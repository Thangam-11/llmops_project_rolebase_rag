from pathlib import Path
from src.data_ingestion.data_loader import DocumentLoader
from src.data_ingestion.chunker_service import ChunkingService

loader  = DocumentLoader()
chunker = ChunkingService()

documents = loader.ingest_directory(Path("data"))

print(f"Total documents loaded: {len(documents)}")

for doc in documents:
    chunks = chunker.chunk_document(
        document  = doc["document"],
        file_type = doc["file_type"],
    )

    print("=" * 80)
    print(f"Department  : {doc['department']}")
    print(f"Filename    : {doc['filename']}")
    print(f"File Type   : {doc['file_type']}")
    print(f"Total Chunks: {len(chunks)}")
    print()

    for chunk in chunks:
        print(f"  chunk_id   : {chunk['chunk_id']}")
        print(f"  chunk_type : {chunk['chunk_type']}")
        print(f"  page       : {chunk['page']}")
        print(f"  headings   : {chunk['headings']}")
        print(f"  text length: {len(chunk['text'])} chars")
        print(f"  text preview:")
        print(f"  {chunk['text'][:200]}")
        print()