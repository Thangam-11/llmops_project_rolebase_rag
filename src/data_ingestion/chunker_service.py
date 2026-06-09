from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


class ChunkingService:

    def __init__(self):

        self.chunker = HybridChunker(
            tokenizer="BAAI/bge-base-en-v1.5",
            max_tokens=512,
            min_tokens=64,
            merge_peers=True,
        )

        logger.info(
            "Chunking Service Initialized"
        )

    def chunk_document(
        self,
        document,
        file_type: str,
    ) -> list[dict]:

        try:

            if file_type == ".csv":

                logger.info(
                    "Using CSV Chunker"
                )

                return self.chunk_csv(
                    document
                )

            logger.info(
                "Using HybridChunker"
            )

            return self.chunk_hybrid(
                document
            )

        except Exception:

            logger.exception(
                "Chunking failed"
            )

            raise

    def chunk_hybrid(
        self,
        document,
    ) -> list[dict]:

        chunks = []

        for idx, chunk in enumerate(
            self.chunker.chunk(document)
        ):

            text = chunk.text.strip()

            if not text:
                continue

            heading_objs = (
                getattr(
                    chunk.meta,
                    "headings",
                    None
                )
                or []
            )

            headings = [
                h.text
                for h in heading_objs
                if hasattr(h, "text")
            ]

            page_number = None

            doc_items = (
                getattr(
                    chunk.meta,
                    "doc_items",
                    None
                )
                or []
            )

            if doc_items:

                first_item = doc_items[0]

                prov = getattr(
                    first_item,
                    "prov",
                    None
                )

                if prov:
                    page_number = (
                        prov[0].page_no
                    )

            chunks.append(
                {
                    "chunk_id": idx,
                    "text": text,
                    "headings": headings,
                    "page": page_number,
                    "chunk_type": "text",
                }
            )

        logger.info(
            f"Generated {len(chunks)} chunks"
        )

        return chunks

    def chunk_csv(
        self,
        document,
        rows_per_chunk: int = 100,
    ) -> list[dict]:

        markdown = (
            document.export_to_markdown()
        )

        lines = markdown.splitlines()

        chunks = []

        for idx, i in enumerate(
            range(
                0,
                len(lines),
                rows_per_chunk,
            )
        ):

            chunk_text = "\n".join(
                lines[
                    i : i + rows_per_chunk
                ]
            )

            chunks.append(
                {
                    "chunk_id": idx,
                    "text": chunk_text,
                    "headings": [],
                    "page": None,
                    "chunk_type": "table",
                }
            )

        logger.info(
            f"Generated {len(chunks)} CSV chunks"
        )

        return chunks