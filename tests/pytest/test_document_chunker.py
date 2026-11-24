"""Real recursive splitting preserves readable content and original provenance."""

from io import BytesIO
import json
from uuid import uuid4

from docx import Document
import pytest

from local_rag.core.config import DocumentSettings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.models import Extraction, ReadDocument, TextUnit
from local_rag.documents.reader import DocumentReader
from local_rag.documents.splitters import DocumentChunker, split_document
from tests.fixtures.reader_documents import unicode_pdf_bytes


@pytest.fixture
def reader(isolated_root):
    return DocumentReader(DocumentSettings(storage_dir=isolated_root / "unused-store"))


def write_text(root, name, text):
    source = root / name
    source.write_text(text, encoding="utf-8")
    return source


def assert_chunk_sources(document, result, size, overlap):
    """Use the reading result as the independent source/position oracle."""
    reading = document.to_dict()
    data = result.to_dict()
    for name in ("document_id", "filename", "file_type", "size_bytes", "page_count", "text", "character_count", "warnings"):
        assert data[name] == reading[name]
    assert data["chunk_config"] == {"chunk_size": size, "chunk_overlap": overlap}
    assert data["chunk_count"] == len(data["chunks"]) > 0
    assert json.loads(json.dumps(data, ensure_ascii=False)) == data
    covered = bytearray(len(data["text"]))
    starts = []
    for index, chunk in enumerate(data["chunks"], 1):
        metadata = chunk["metadata"]
        start, end = metadata["start_index"], metadata["end_index"]
        assert 0 <= start < end <= len(data["text"])
        assert data["text"][start:end] == chunk["text"]
        assert 0 < len(chunk["text"]) <= size
        assert chunk["text"].strip()
        assert chunk["chunk_id"] == f"{reading['document_id']}:{index}"
        assert metadata["chunk_index"] == index
        assert metadata["document_id"] == reading["document_id"]
        assert metadata["filename"] == reading["filename"]
        assert metadata["file_type"] == reading["file_type"]
        unit = reading["units"][metadata["unit_index"] - 1]["metadata"]
        assert unit["start_index"] <= start < end <= unit["end_index"]
        for field in ("page_number", "section_title", "heading_level"):
            assert metadata[field] == unit[field]
        paragraphs = [p for p in unit["paragraphs"] if p["start_index"] < end and p["end_index"] > start]
        assert metadata["paragraphs"] == paragraphs
        assert metadata["paragraph_start"] == (paragraphs[0]["paragraph_index"] if paragraphs else None)
        assert metadata["paragraph_end"] == (paragraphs[-1]["paragraph_index"] if paragraphs else None)
        covered[start:end] = b"\x01" * (end - start)
        starts.append(start)
    assert all(left < right for left, right in zip(starts, starts[1:]))
    assert all(covered[index] or char.isspace() for index, char in enumerate(data["text"]))
    return data


@pytest.mark.parametrize("size,overlap,length,expected_starts,expected_lengths", [
    (500, 80, 1603, [0, 420, 840, 1260], [500, 500, 500, 343]),
    (700, 100, 2203, [0, 600, 1200, 1800], [700, 700, 700, 403]),
    (800, 120, 2503, [0, 680, 1360, 2040], [800, 800, 800, 463]),
])
def test_long_chinese_fallback_has_exact_character_lengths_overlap_and_tail(
        reader, isolated_root, size, overlap, length, expected_starts, expected_lengths):
    # Distinct CJK characters have no selected separator; repeated bytes cannot
    # accidentally satisfy the source position and overlap assertions.
    text = "".join(chr(0x4E00 + index) for index in range(length))
    source = write_text(isolated_root, "无分隔符长文.txt", text)
    document = reader.read(source)
    before = document.to_dict()
    data = assert_chunk_sources(document, DocumentChunker(size, overlap).split(document), size, overlap)
    assert [chunk["metadata"]["start_index"] for chunk in data["chunks"]] == expected_starts
    assert [len(chunk["text"]) for chunk in data["chunks"]] == expected_lengths
    for previous, current in zip(data["chunks"], data["chunks"][1:]):
        assert previous["metadata"]["end_index"] - current["metadata"]["start_index"] == overlap
        assert previous["text"][-overlap:] == current["text"][:overlap]
    assert document.to_dict() == before
    assert source.read_text(encoding="utf-8") == text
    assert list(isolated_root.iterdir()) == [source]


def test_defaults_and_short_document_preserve_single_chunk(reader, isolated_root):
    source = write_text(isolated_root, "短文.txt", "青岚实验室每周二和周四接收样品。")
    document = reader.read(source)
    data = assert_chunk_sources(document, DocumentChunker().split(document), 700, 100)
    assert data["chunk_count"] == 1
    assert data["chunks"][0]["text"] == document.text
    assert data["chunks"][0]["metadata"]["page_number"] is None


def test_zero_overlap_is_supported_without_losing_characters(reader, isolated_root):
    text = "".join(chr(0x4E00 + index) for index in range(1203))
    document = reader.read(write_text(isolated_root, "无重叠.txt", text))
    data = assert_chunk_sources(document, DocumentChunker(500, 0).split(document), 500, 0)
    assert [len(chunk["text"]) for chunk in data["chunks"]] == [500, 500, 203]
    assert "".join(chunk["text"] for chunk in data["chunks"]) == text


def test_repeated_chinese_paragraphs_keep_order_and_cover_all_non_whitespace(reader, isolated_root):
    repeated = "样品接收需要登记编号，完成后放入冷藏区。登记前应核对标签！"
    text = "\n\n".join(f"第{number:03d}段：" + repeated * 8 for number in range(25))
    source = write_text(isolated_root, "重复段落.txt", text)
    document = reader.read(source)
    data = assert_chunk_sources(document, DocumentChunker(500, 80).split(document), 500, 80)
    assert data["chunk_count"] > 5
    # Complete identical repeated paragraphs exercise LangChain's offset lookup,
    # rather than relying only on content that occurs exactly once.
    identical = reader.read(write_text(isolated_root, "重复原文.txt", (repeated + "\n\n") * 100))
    identical_data = assert_chunk_sources(identical, DocumentChunker().split(identical), 700, 100)
    assert identical_data["chunk_count"] > 1


def test_pdf_never_crosses_pages_and_keeps_third_physical_page(reader, isolated_root):
    source = isolated_root / "分页资料.pdf"
    source.write_bytes(unicode_pdf_bytes(["第一页甲乙丙丁" * 180, "", "第三页戊己庚辛" * 170]))
    document = reader.read(source)
    data = assert_chunk_sources(document, DocumentChunker(500, 80).split(document), 500, 80)
    assert data["page_count"] == 3
    assert any("第 2 页" in message for message in data["warnings"])
    assert {chunk["metadata"]["page_number"] for chunk in data["chunks"]} == {1, 3}
    assert {chunk["metadata"]["unit_index"] for chunk in data["chunks"]} == {1, 2}
    for chunk in data["chunks"]:
        forbidden = "戊" if chunk["metadata"]["page_number"] == 1 else "甲"
        assert forbidden not in chunk["text"]


@pytest.mark.parametrize("extension", ["docx", "md"])
def test_word_and_markdown_keep_section_and_paragraph_positions(reader, isolated_root, extension):
    first, second = "样品登记编号，接收之后冷藏保存。" * 70, "设备归还前记录检测结果。" * 75
    source = isolated_root / f"章节资料.{extension}"
    if extension == "docx":
        word = Document()
        word.add_heading("样品接收", 1)
        word.add_paragraph(first)
        table = word.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "联系人"
        table.cell(0, 1).text = "林杉"
        word.add_heading("设备归还", 2)
        word.add_paragraph(second)
        stream = BytesIO()
        word.save(stream)
        source.write_bytes(stream.getvalue())
    else:
        source.write_text(f"# 样品接收\n\n{first}\n\n联系人：林杉。\n\n## 设备归还\n\n{second}\n", encoding="utf-8")
    document = reader.read(source)
    before = document.to_dict()
    data = assert_chunk_sources(document, DocumentChunker(500, 80).split(document), 500, 80)
    assert data["chunk_count"] > 4
    assert {chunk["metadata"]["section_title"] for chunk in data["chunks"]} == {"样品接收", "设备归还"}
    assert {chunk["metadata"]["heading_level"] for chunk in data["chunks"]} == {1, 2}
    assert all(chunk["metadata"]["page_number"] is None for chunk in data["chunks"])
    assert all(chunk["metadata"]["paragraphs"] for chunk in data["chunks"])
    for chunk in data["chunks"]:
        forbidden = "设备归还" if chunk["metadata"]["section_title"] == "样品接收" else "样品登记"
        assert forbidden not in chunk["text"]
    assert document.to_dict() == before


def test_same_read_document_is_deterministic_and_other_document_ids_never_collide(reader, isolated_root):
    source = write_text(isolated_root, "来源.txt", "记录编号以供核对来源。" * 180)
    first = reader.read(source)
    second = reader.read(source)
    chunker = DocumentChunker()
    baseline = chunker.split(first).to_dict()
    assert chunker.split(first).to_dict() == baseline
    other = chunker.split(second).to_dict()
    assert first.document_id != second.document_id
    assert {c["chunk_id"] for c in baseline["chunks"]}.isdisjoint(c["chunk_id"] for c in other["chunks"])
    assert [c["text"] for c in baseline["chunks"]] == [c["text"] for c in other["chunks"]]


def test_legacy_split_document_uses_the_same_source_contract(reader, isolated_root):
    document = reader.read(write_text(isolated_root, "兼容入口.txt", "来源映射保持一致。" * 150))
    data = DocumentChunker(800, 120).split(document).to_dict()
    text, chunks = split_document(document.extraction, document.document_id, document.filename,
                                  ".txt", chunk_size=800, chunk_overlap=120)
    assert text == data["text"]
    assert chunks == data["chunks"]


@pytest.mark.parametrize("size,overlap", [
    (49, 0), (10001, 100), (700, -1), (700, 700), (700, 701),
    (True, 0), (700, True), (700.0, 100), (700, 100.0), ("700", 100),
])
def test_invalid_chunk_configuration_fails_before_splitting(size, overlap):
    with pytest.raises(ValueError):
        DocumentChunker(size, overlap)
    with pytest.raises(ValueError):
        split_document(Extraction([TextUnit("正常文本。")]), str(uuid4()), "sample.txt", ".txt", size, overlap)


@pytest.mark.parametrize("units", [[], [TextUnit("")], [TextUnit(" \n\t ")]])
def test_empty_reading_result_is_rejected_with_clear_error(units):
    document = ReadDocument(document_id=str(uuid4()), filename="empty.txt", file_type="txt",
                            size_bytes=0, extraction=Extraction(units))
    with pytest.raises(DocumentError) as error:
        DocumentChunker().split(document)
    assert error.value.code == "no_extractable_text"
    assert error.value.message
