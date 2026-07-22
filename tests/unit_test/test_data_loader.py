# tests/unit/test_data_loader.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data_ingestion.data_loader import (
    IngestionService,
    FOLDER_TO_DEPT,
    VALID_DEPARTMENTS,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    """IngestionService with a mocked Docling converter — no real parsing."""
    svc = IngestionService()
    svc.converter = MagicMock()
    return svc


def make_docling_result(markdown_text: str):
    """Fake the object returned by converter.convert(path)."""
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = markdown_text
    mock_result = MagicMock()
    mock_result.document = mock_doc
    return mock_result


# ---------------------------------------------------------------------------
# 1. FOLDER_TO_DEPT mapping — lock in business logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder,expected_dept", [
    ("engineering", "engineering"),
    ("finance", "finance"),
    ("marketing", "marketing"),
    ("general", "general"),
    ("hr", "hr"),
    ("hr_data", "hr"),          # the non-obvious alias — most important case
    ("c_level", "c_level"),
])
def test_folder_to_dept_mapping(folder, expected_dept):
    assert FOLDER_TO_DEPT[folder] == expected_dept


def test_unknown_folder_not_in_mapping():
    assert "random_folder" not in FOLDER_TO_DEPT


def test_valid_departments_has_no_folder_aliases():
    # "hr_data" is a folder name, not a department — must not leak into VALID_DEPARTMENTS
    assert "hr_data" not in VALID_DEPARTMENTS
    assert "hr" in VALID_DEPARTMENTS


# ---------------------------------------------------------------------------
# 2. ingest_file
# ---------------------------------------------------------------------------

def test_ingest_file_success(service, tmp_path):
    fake_file = tmp_path / "report.pdf"
    fake_file.write_text("dummy")
    service.converter.convert.return_value = make_docling_result("# hello")

    result = service.ingest_file(fake_file, department="finance")

    assert result["department"] == "finance"
    assert result["filename"] == "report.pdf"
    assert result["file_type"] == ".pdf"
    assert result["file_path"] == str(fake_file)
    service.converter.convert.assert_called_once_with(str(fake_file))


def test_ingest_file_raises_on_conversion_failure(service, tmp_path):
    fake_file = tmp_path / "broken.pdf"
    fake_file.write_text("dummy")
    service.converter.convert.side_effect = RuntimeError("Docling parse error")

    with pytest.raises(RuntimeError):
        service.ingest_file(fake_file, department="finance")


# ---------------------------------------------------------------------------
# 3. ingest_directory — the skip logic
# ---------------------------------------------------------------------------

def test_ingest_directory_missing_root_returns_empty(service, tmp_path):
    missing = tmp_path / "does_not_exist"
    assert service.ingest_directory(missing) == []


def test_ingest_directory_skips_unknown_folder(service, tmp_path):
    unknown = tmp_path / "random_dept"
    unknown.mkdir()
    (unknown / "file.pdf").write_text("x")

    result = service.ingest_directory(tmp_path)

    assert result == []
    service.converter.convert.assert_not_called()


def test_ingest_directory_skips_unsupported_extension(service, tmp_path):
    dept_dir = tmp_path / "finance"
    dept_dir.mkdir()
    (dept_dir / "notes.exe").write_text("x")   # not in SUPPORTED_EXTENSIONS

    result = service.ingest_directory(tmp_path)

    assert result == []
    service.converter.convert.assert_not_called()


def test_ingest_directory_maps_hr_data_folder_to_hr_department(service, tmp_path):
    dept_dir = tmp_path / "hr_data"
    dept_dir.mkdir()
    (dept_dir / "policy.pdf").write_text("x")
    service.converter.convert.return_value = make_docling_result("policy text")

    result = service.ingest_directory(tmp_path)

    assert len(result) == 1
    assert result[0]["department"] == "hr"     # not "hr_data"


def test_ingest_directory_continues_after_single_file_failure(service, tmp_path):
    dept_dir = tmp_path / "finance"
    dept_dir.mkdir()
    (dept_dir / "good.pdf").write_text("x")
    (dept_dir / "bad.pdf").write_text("x")

    # first call fails, second succeeds — order depends on sorted() rglob
    service.converter.convert.side_effect = [
        RuntimeError("boom"),
        make_docling_result("ok text"),
    ]

    result = service.ingest_directory(tmp_path)

    assert len(result) == 1   # one file skipped, one succeeded, no crash


def test_ingest_directory_ignores_non_directory_entries(service, tmp_path):
    (tmp_path / "stray_file.txt").write_text("x")   # sibling file, not a dept folder

    result = service.ingest_directory(tmp_path)

    assert result == []


# ---------------------------------------------------------------------------
# 4. to_langchain_docs — metadata + empty-content handling
# ---------------------------------------------------------------------------

def _raw_doc(text: str, filename="f.pdf", department="finance", file_path="/tmp/f.pdf"):
    return {
        "department": department,
        "filename": filename,
        "file_path": file_path,
        "file_type": ".pdf",
        "document": make_docling_result(text).document,
    }


def test_to_langchain_docs_sets_correct_metadata(service):
    raw = [_raw_doc("some content here")]

    docs = service.to_langchain_docs(raw)

    assert len(docs) == 1
    assert docs[0].page_content == "some content here"
    assert docs[0].metadata == {
        "source": "/tmp/f.pdf",
        "filename": "f.pdf",
        "department": "finance",
        "file_type": ".pdf",
    }


def test_to_langchain_docs_drops_empty_content(service):
    raw = [_raw_doc("   ")]   # whitespace-only

    docs = service.to_langchain_docs(raw)

    assert docs == []


def test_to_langchain_docs_skips_doc_on_export_failure(service):
    raw = [_raw_doc("good")]
    raw[0]["document"].export_to_markdown.side_effect = RuntimeError("boom")

    docs = service.to_langchain_docs(raw)

    assert docs == []   # failure on one doc doesn't crash the batch


def test_to_langchain_docs_partial_batch_success(service):
    good = _raw_doc("good content", filename="good.pdf")
    bad = _raw_doc("irrelevant", filename="bad.pdf")
    bad["document"].export_to_markdown.side_effect = RuntimeError("boom")

    docs = service.to_langchain_docs([good, bad])

    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "good.pdf"


# ---------------------------------------------------------------------------
# 5. Convenience wrappers — thin, but worth a smoke test
# ---------------------------------------------------------------------------

def test_load_file_returns_langchain_document(service, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("x")
    service.converter.convert.return_value = make_docling_result("content")

    docs = service.load_file(f, department="engineering")

    assert len(docs) == 1
    assert docs[0].metadata["department"] == "engineering"


def test_load_directory_end_to_end_with_mocked_converter(service, tmp_path):
    dept_dir = tmp_path / "marketing"
    dept_dir.mkdir()
    (dept_dir / "brief.pdf").write_text("x")
    service.converter.convert.return_value = make_docling_result("marketing content")

    docs = service.load_directory(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["department"] == "marketing"