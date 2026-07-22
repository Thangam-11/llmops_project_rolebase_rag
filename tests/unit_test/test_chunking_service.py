# tests/unit/test_chunking_service.py
from unittest.mock import MagicMock

import pytest

from src.data_ingestion.chunker_service import ChunkingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service(mocker):
    """
    Patch HybridChunker where it's imported in chunking_service, so __init__
    never tries to download BAAI/bge-base-en-v1.5 from HuggingFace.
    """
    mock_chunker_cls = mocker.patch(
        "src.data_ingestion.chunker_service.HybridChunker"
    )
    mock_chunker_instance = MagicMock()
    mock_chunker_cls.return_value = mock_chunker_instance

    svc = ChunkingService()
    svc.chunker = mock_chunker_instance  # convenience alias for tests
    return svc


def make_raw_chunk(text, headings=None, page_no=None, headings_attr_missing=False,
                    doc_items_attr_missing=False, prov_missing=False):
    """
    Build a fake object mimicking what self.chunker.chunk(document) yields —
    matching the exact getattr access pattern in chunk_hybrid.
    """
    chunk = MagicMock()
    chunk.text = text

    if headings_attr_missing:
        del chunk.meta.headings  # getattr(..., "headings", None) -> None via AttributeError path
        # MagicMock auto-creates attrs, so simulate "missing" by setting to None directly instead:
    chunk.meta.headings = None if headings_attr_missing else (headings or [])

    if headings:
        heading_objs = []
        for h_text in headings:
            h = MagicMock()
            h.text = h_text
            heading_objs.append(h)
        chunk.meta.headings = heading_objs

    if doc_items_attr_missing:
        chunk.meta.doc_items = None
    elif page_no is not None:
        item = MagicMock()
        if prov_missing:
            item.prov = None
        else:
            prov_entry = MagicMock()
            prov_entry.page_no = page_no
            item.prov = [prov_entry]
        chunk.meta.doc_items = [item]
    else:
        chunk.meta.doc_items = []

    return chunk


@pytest.fixture
def fake_document():
    return MagicMock()


# ---------------------------------------------------------------------------
# 1. chunk_document routing
# ---------------------------------------------------------------------------

def test_chunk_document_routes_csv_to_chunk_csv(service, fake_document, mocker):
    spy = mocker.patch.object(service, "chunk_csv", return_value=[{"chunk_id": 0}])
    result = service.chunk_document(fake_document, file_type=".csv")

    spy.assert_called_once_with(fake_document)
    assert result == [{"chunk_id": 0}]


@pytest.mark.parametrize("file_type", [".pdf", ".docx", ".txt", ".md", ".pptx"])
def test_chunk_document_routes_non_csv_to_chunk_hybrid(service, fake_document, mocker, file_type):
    spy = mocker.patch.object(service, "chunk_hybrid", return_value=[{"chunk_id": 0}])
    result = service.chunk_document(fake_document, file_type=file_type)

    spy.assert_called_once_with(fake_document)
    assert result == [{"chunk_id": 0}]


def test_chunk_document_reraises_on_failure(service, fake_document, mocker):
    mocker.patch.object(service, "chunk_hybrid", side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        service.chunk_document(fake_document, file_type=".pdf")


# ---------------------------------------------------------------------------
# 2. chunk_hybrid — text/empty handling
# ---------------------------------------------------------------------------

def test_chunk_hybrid_skips_empty_text(service, fake_document):
    service.chunker.chunk.return_value = [
        make_raw_chunk("real content"),
        make_raw_chunk("   "),          # whitespace only -> skipped
        make_raw_chunk(""),             # empty -> skipped
    ]

    result = service.chunk_hybrid(fake_document)

    assert len(result) == 1
    assert result[0]["text"] == "real content"


def test_chunk_hybrid_strips_whitespace_from_text(service, fake_document):
    service.chunker.chunk.return_value = [make_raw_chunk("  hello world  ")]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["text"] == "hello world"


def test_chunk_hybrid_returns_empty_list_when_no_chunks(service, fake_document):
    service.chunker.chunk.return_value = []
    assert service.chunk_hybrid(fake_document) == []


# ---------------------------------------------------------------------------
# 3. chunk_hybrid — headings extraction
# ---------------------------------------------------------------------------

def test_chunk_hybrid_extracts_headings(service, fake_document):
    service.chunker.chunk.return_value = [
        make_raw_chunk("content", headings=["Section 1", "Subsection A"])
    ]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["headings"] == ["Section 1", "Subsection A"]


def test_chunk_hybrid_headings_none_becomes_empty_list(service, fake_document):
    service.chunker.chunk.return_value = [
        make_raw_chunk("content", headings_attr_missing=True)
    ]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["headings"] == []


def test_chunk_hybrid_headings_empty_list_stays_empty(service, fake_document):
    service.chunker.chunk.return_value = [make_raw_chunk("content", headings=[])]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["headings"] == []


# ---------------------------------------------------------------------------
# 4. chunk_hybrid — page number extraction (the risky nested getattr chain)
# ---------------------------------------------------------------------------

def test_chunk_hybrid_extracts_page_number(service, fake_document):
    service.chunker.chunk.return_value = [make_raw_chunk("content", page_no=7)]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["page"] == 7


def test_chunk_hybrid_page_none_when_no_doc_items(service, fake_document):
    service.chunker.chunk.return_value = [make_raw_chunk("content", doc_items_attr_missing=True)]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["page"] is None


def test_chunk_hybrid_page_none_when_doc_items_empty(service, fake_document):
    chunk = make_raw_chunk("content")
    chunk.meta.doc_items = []  # explicitly empty, not None

    service.chunker.chunk.return_value = [chunk]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["page"] is None


def test_chunk_hybrid_page_none_when_prov_missing(service, fake_document):
    chunk = make_raw_chunk("content", page_no=3, prov_missing=True)

    service.chunker.chunk.return_value = [chunk]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["page"] is None


# ---------------------------------------------------------------------------
# 5. chunk_hybrid — the chunk_id gap behavior (document current behavior explicitly)
# ---------------------------------------------------------------------------

def test_chunk_hybrid_chunk_id_reflects_original_enumerate_index_not_renumbered(service, fake_document):
    """
    idx comes from enumerate() BEFORE the empty-text `continue`.
    So if the 2nd raw chunk (index 1) is empty and skipped, the 3rd raw chunk
    keeps chunk_id=2, not renumbered to 1. This test locks in current behavior —
    if this is not the desired behavior, fix the code, then update this test.
    """
    service.chunker.chunk.return_value = [
        make_raw_chunk("first"),     # idx 0
        make_raw_chunk("   "),       # idx 1 -> skipped (empty)
        make_raw_chunk("third"),     # idx 2
    ]

    result = service.chunk_hybrid(fake_document)

    assert len(result) == 2
    assert result[0]["chunk_id"] == 0
    assert result[1]["chunk_id"] == 2   # gap, not 1 — confirms current behavior


def test_chunk_hybrid_sets_chunk_type_text(service, fake_document):
    service.chunker.chunk.return_value = [make_raw_chunk("content")]

    result = service.chunk_hybrid(fake_document)

    assert result[0]["chunk_type"] == "text"


# ---------------------------------------------------------------------------
# 6. chunk_csv
# ---------------------------------------------------------------------------

def test_chunk_csv_groups_rows_correctly(service):
    fake_doc = MagicMock()
    lines = [f"row{i}" for i in range(250)]
    fake_doc.export_to_markdown.return_value = "\n".join(lines)

    result = service.chunk_csv(fake_doc, rows_per_chunk=100)

    assert len(result) == 3   # 100, 100, 50
    assert result[0]["text"].splitlines() == lines[0:100]
    assert result[1]["text"].splitlines() == lines[100:200]
    assert result[2]["text"].splitlines() == lines[200:250]


def test_chunk_csv_empty_document_returns_no_chunks(service):
    fake_doc = MagicMock()
    fake_doc.export_to_markdown.return_value = ""

    result = service.chunk_csv(fake_doc)

    assert result == []


def test_chunk_csv_sets_metadata_defaults(service):
    fake_doc = MagicMock()
    fake_doc.export_to_markdown.return_value = "row1\nrow2"

    result = service.chunk_csv(fake_doc)

    assert result[0]["chunk_type"] == "table"
    assert result[0]["headings"] == []
    assert result[0]["page"] is None
    assert result[0]["chunk_id"] == 0


def test_chunk_csv_single_chunk_when_fewer_rows_than_chunk_size(service):
    fake_doc = MagicMock()
    fake_doc.export_to_markdown.return_value = "row1\nrow2\nrow3"

    result = service.chunk_csv(fake_doc, rows_per_chunk=100)

    assert len(result) == 1
    assert result[0]["text"] == "row1\nrow2\nrow3"