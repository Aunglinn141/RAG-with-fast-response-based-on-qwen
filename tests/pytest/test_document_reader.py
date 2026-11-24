"""Standalone reading uses real files/parsers without API, models or storage writes."""

from io import BytesIO
import json
from uuid import UUID, uuid4

from docx import Document
from pypdf import PdfReader
import pytest

from local_rag.core.config import DocumentSettings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.reader import DocumentReader
from tests.fixtures.documents import pdf_bytes
from tests.fixtures.reader_documents import image_only_pdf_bytes, unicode_pdf_bytes


@pytest.fixture
def reader(isolated_root):
    return DocumentReader(DocumentSettings(storage_dir=isolated_root / "unused-document-store", ocr_mode="off"))


def write_source(root, filename, body):
    path = root / filename
    path.write_bytes(body)
    return path


def assert_source_metadata(result, expected_filename, expected_type):
    record = result.to_dict()
    assert str(UUID(record["document_id"])) == record["document_id"]
    assert record["filename"] == expected_filename
    assert record["file_type"] == expected_type
    assert record["text"] == result.text
    assert record["character_count"] == len(record["text"])
    assert record["units"]
    assert record["text"] == "\n\n".join(unit["text"] for unit in record["units"])
    assert len(record["units"]) == len(result.extraction.units)
    for index, unit in enumerate(record["units"], 1):
        source = unit["metadata"]
        assert source["document_id"] == record["document_id"]
        assert source["filename"] == expected_filename
        assert source["file_type"] == expected_type
        assert source["unit_index"] == index
        assert "page_number" in source
        assert record["text"][source["start_index"]:source["end_index"]] == unit["text"]
        for paragraph in source["paragraphs"]:
            assert source["start_index"] <= paragraph["start_index"] < paragraph["end_index"] <= source["end_index"]
            assert record["text"][paragraph["start_index"]:paragraph["end_index"]].strip()
    # JSON output must contain plain values, not Path or dataclass instances.
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record
    return record


def test_chinese_pdf_text_layer_preserves_physical_pages_and_blank_page_warning(reader, isolated_root):
    source = write_source(isolated_root, "中文资料.PDF", unicode_pdf_bytes([
        "第一页：青岚实验室每周二开放。", "", "第三页：负责人是林杉。",
    ]))
    result = reader.read(source)
    record = assert_source_metadata(result, source.name, "pdf")
    assert record["page_count"] == 3
    assert [unit["metadata"]["page_number"] for unit in record["units"]] == [1, 3]
    assert [unit["text"] for unit in record["units"]] == [
        "第一页：青岚实验室每周二开放。", "第三页：负责人是林杉。",
    ]
    assert any("第 2 页" in warning and "OCR" in warning for warning in record["warnings"])
    assert record["size_bytes"] == source.stat().st_size


def test_real_image_only_pdf_returns_no_text_error_without_ocr(reader, isolated_root):
    body = image_only_pdf_bytes()
    page = PdfReader(BytesIO(body)).pages[0]
    assert page.extract_text() == ""
    assert page["/Resources"]["/XObject"]["/Im0"]["/Subtype"] == "/Image"
    source = write_source(isolated_root, "扫描样例.pdf", body)
    with pytest.raises(DocumentError) as error:
        reader.read(source)
    assert error.value.code == "no_extractable_text"
    assert error.value.status == 422
    assert "OCR" in error.value.message
    assert source.read_bytes() == body


def test_docx_preserves_chinese_heading_paragraphs_and_table_read_order(reader, isolated_root):
    document = Document()
    document.add_heading("项目说明", 0)
    document.add_heading("样品接收", 1)
    document.add_paragraph("每周二和周四开放。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "联系人"
    table.cell(0, 1).text = "林杉"
    document.add_heading("归档规则", 2)
    document.add_paragraph("使用项目编号归档。")
    data = BytesIO()
    document.save(data)
    source = write_source(isolated_root, "读取规范.docx", data.getvalue())
    record = assert_source_metadata(reader.read(source), source.name, "docx")
    assert record["page_count"] is None
    assert all(unit["metadata"]["page_number"] is None for unit in record["units"])
    assert [(unit["metadata"]["section_title"], unit["metadata"]["heading_level"])
            for unit in record["units"]] == [("项目说明", 0), ("样品接收", 1), ("归档规则", 2)]
    assert "每周二和周四开放。\n\n联系人\n\n林杉" in record["text"]
    indices = [paragraph["paragraph_index"] for unit in record["units"]
               for paragraph in unit["metadata"]["paragraphs"]]
    assert indices == list(range(1, 8))
    for unit in record["units"]:
        metadata = unit["metadata"]
        assert metadata["paragraph_start"] == metadata["paragraphs"][0]["paragraph_index"]
        assert metadata["paragraph_end"] == metadata["paragraphs"][-1]["paragraph_index"]


def test_markdown_returns_plain_text_and_heading_metadata(reader, isolated_root):
    body = "# 概览\n\n请阅读 **重点** 与 [详细说明](https://example.test/read)。\n\n## 操作\n\n- 先上传文档\n- 再核对来源\n\n```python\nprint('中文')\n```\n"
    source = write_source(isolated_root, "使用说明.md", body.encode("utf-8"))
    record = assert_source_metadata(reader.read(source), source.name, "md")
    assert "请阅读 重点 与 详细说明。" in record["text"]
    assert "先上传文档" in record["text"] and "再核对来源" in record["text"]
    assert "print('中文')" in record["text"]
    assert "```" not in record["text"] and "https://" not in record["text"]
    assert [unit["metadata"]["section_title"] for unit in record["units"]] == ["概览", "操作"]
    assert [unit["metadata"]["heading_level"] for unit in record["units"]] == [1, 2]
    assert record["page_count"] is None
    assert all(unit["metadata"]["page_number"] is None for unit in record["units"])


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32", "gb18030"])
def test_txt_chinese_encodings_and_whitespace(reader, isolated_root, encoding):
    body = "  中文\t   文档。\r\n\r\n\r\n第二段内容。  ".encode(encoding)
    source = write_source(isolated_root, "文字.TXT", body)
    record = assert_source_metadata(reader.read(source), source.name, "txt")
    assert record["text"] == "中文 文档。\n\n第二段内容。"
    assert record["page_count"] is None
    assert record["units"][0]["metadata"]["page_number"] is None
    assert record["warnings"] == []


def test_read_only_operation_generates_independent_ids_and_preserves_supplied_id(reader, isolated_root):
    source = write_source(isolated_root, "original.txt", "原文件不变。".encode("utf-8"))
    before = source.read_bytes()
    first, second = reader.read(source).to_dict(), reader.read(source).to_dict()
    assert first["document_id"] != second["document_id"]
    supplied = str(uuid4())
    named = reader.read(source, document_id=supplied.upper(), filename="C:\\fakepath\\来源.txt").to_dict()
    assert named["document_id"] == supplied
    assert named["filename"] == "来源.txt"
    assert named["units"][0]["metadata"]["filename"] == "来源.txt"
    assert named["units"][0]["metadata"]["document_id"] == supplied
    assert source.read_bytes() == before
    assert list(isolated_root.iterdir()) == [source]


@pytest.mark.parametrize("filename,body,status,code", [
    ("empty.txt", b"", 422, "empty_file"),
    ("blank.txt", b" \n\t\r\n", 422, "no_extractable_text"),
    ("program.exe", b"unsupported", 415, "unsupported_file_type"),
    ("old.doc", b"unsupported", 415, "unsupported_file_type"),
    ("fake.pdf", b"plain text", 415, "file_type_mismatch"),
    ("broken.pdf", b"%PDF-1.7\nnot a PDF structure", 422, "invalid_document"),
    ("binary.txt", b"\x00\x01binary", 415, "binary_text_file"),
])
def test_invalid_files_return_clear_errors(reader, isolated_root, filename, body, status, code):
    source = write_source(isolated_root, filename, body)
    with pytest.raises(DocumentError) as error:
        reader.read(source)
    assert (error.value.status, error.value.code) == (status, code)
    assert error.value.message
    assert source.read_bytes() == body


def test_encrypted_pdf_returns_clear_error(reader, isolated_root):
    source = write_source(isolated_root, "locked.pdf", pdf_bytes(["Private text"], password="secret"))
    with pytest.raises(DocumentError) as error:
        reader.read(source)
    assert (error.value.status, error.value.code) == (422, "encrypted_pdf")


def test_missing_file_and_directory_are_distinguished(reader, isolated_root):
    with pytest.raises(DocumentError) as error:
        reader.read(isolated_root / "missing.txt")
    assert (error.value.status, error.value.code) == (404, "document_file_not_found")
    directory = isolated_root / "directory.txt"
    directory.mkdir()
    with pytest.raises(DocumentError) as error:
        reader.read(directory)
    assert (error.value.status, error.value.code) == (400, "invalid_document_path")


def test_invalid_supplied_document_id_is_rejected(reader, isolated_root):
    source = write_source(isolated_root, "valid.txt", b"text")
    with pytest.raises(DocumentError) as error:
        reader.read(source, document_id="not-a-uuid")
    assert (error.value.status, error.value.code) == (400, "invalid_document_id")


def test_reader_enforces_file_size_exact_boundary_and_text_budget(isolated_root):
    settings = DocumentSettings(storage_dir=isolated_root / "unused", max_file_bytes=16)
    reader = DocumentReader(settings)
    source = write_source(isolated_root, "boundary.txt", b"x" * 16)
    assert reader.read(source).text == "x" * 16
    source.write_bytes(b"x" * 17)
    with pytest.raises(DocumentError) as error:
        reader.read(source)
    assert (error.value.status, error.value.code) == (413, "file_too_large")
    source.write_bytes(b"abcdef")
    reader = DocumentReader(DocumentSettings(storage_dir=isolated_root / "unused", max_text_characters=5))
    with pytest.raises(DocumentError) as error:
        reader.read(source)
    assert (error.value.status, error.value.code) == (413, "extracted_text_too_large")
