"""Real local OCR/parsers plus upload/chunk/index contracts, isolated from user data."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import shutil
from uuid import uuid4

from fastapi.testclient import TestClient
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
import pytest

from local_rag.core.config import DocumentSettings, VectorSettings, RuntimeSettings, RUNTIME_DIR
from local_rag.core.exceptions import DocumentError
from local_rag.documents.reader import DocumentReader
from local_rag.documents.splitters import DocumentChunker
from local_rag.documents import ocr
from tests.fixtures.documents import pdf_bytes
from tests.fixtures.embeddings import FakeEmbeddings


def picture():
    image = Image.new("RGB", (1000, 300), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    draw.text((30, 25), "LOCAL LIBRARY", fill="black", font=font)
    draw.text((30, 100), "Opening hours 09:00 - 17:00", fill="black", font=font)
    draw.text((30, 175), "Contact Alice", fill="black", font=font)
    return image


def image_bytes(format="PNG", *, multiple=False):
    with picture() as image:
        buffer = BytesIO()
        options = {"save_all": True, "append_images": [image]} if multiple else {}
        image.save(buffer, format=format, **options)
        return buffer.getvalue()


def workbook_bytes():
    book = Workbook()
    first = book.active
    first.title = "人物"
    first.append(["姓名", "经历", "次数"])
    first.append(["孙悟空", "大闹天宫。" * 180, 0])
    first.append([None, None, None])
    first.append(["唐僧", "西天取经", False])
    first["C5"] = "=1+2"
    second = book.create_sheet("地点")
    second.append(["地点", "说明"])
    second.append(["花果山", "孙悟空的家乡"])
    stream = BytesIO()
    book.save(stream)
    book.close()
    return stream.getvalue()


@pytest.fixture(scope="module")
def ocr_config():
    root = RUNTIME_DIR / "tests" / ("ocr-models-" + uuid4().hex)
    settings = DocumentSettings(ocr_model_dir=root / "models")
    ocr.prepare_models(settings.ocr_model_dir)
    yield settings
    with ocr._OCR_LOCK:
        ocr._engine.cache_clear()
    assert root.resolve().parent == (RUNTIME_DIR / "tests").resolve()
    shutil.rmtree(root)


def read_bytes(root, name, body, settings=None):
    path = root / name
    path.write_bytes(body)
    return DocumentReader(settings or DocumentSettings()).read(path)


def assert_chunks(reading):
    result = DocumentChunker(700, 100).split(reading)
    for i, chunk in enumerate(result.chunks, 1):
        meta = chunk["metadata"]
        assert result.text[meta["start_index"]:meta["end_index"]] == chunk["text"]
        assert len(chunk["text"]) <= 700
        assert meta["document_id"] == reading.document_id
        assert meta["filename"] == reading.filename
        assert meta["chunk_index"] == i
    return result


@pytest.mark.parametrize("extension", ["xlsx", "xlsm"])
def test_excel_rows_sheets_formulas_and_chunk_ranges(isolated_root, extension):
    reading = read_bytes(isolated_root, "book." + extension, workbook_bytes())
    result = assert_chunks(reading)
    assert len(result.chunks) > 4
    assert {c["metadata"]["sheet_name"] for c in result.chunks} == {"人物", "地点"}
    assert "0" in result.text and "FALSE" in result.text
    assert "[未计算公式：=1+2]" in result.text
    assert any("1 个公式" in warning for warning in result.warnings)
    for c in result.chunks:
        meta = c["metadata"]
        assert meta["page_number"] is None
        assert meta["row_start"] == min(p["row_index"] for p in meta["paragraphs"])
        assert meta["row_end"] == max(p["row_index"] for p in meta["paragraphs"])
        assert meta["cell_ranges"] == [p["cell_range"] for p in meta["paragraphs"]]
        assert all(p["row_index"] != 3 for p in meta["paragraphs"])
    metadata = reading.to_dict()["units"][0]["metadata"]
    assert metadata["sheet_name"] == "人物" and metadata["row_end"] == 5


def test_xls_actual_legacy_workbook(isolated_root):
    import xlwt
    book = xlwt.Workbook()
    sheet = book.add_sheet("旧表格")
    for r, values in enumerate([["姓名", "事件"], ["关羽", "过五关斩六将"]]):
        for c, value in enumerate(values):
            sheet.write(r, c, value)
    body = BytesIO()
    book.save(body)
    reading = read_bytes(isolated_root, "legacy.xls", body.getvalue())
    assert "过五关斩六将" in reading.text
    assert assert_chunks(reading).chunks[0]["metadata"]["sheet_name"] == "旧表格"


@pytest.mark.parametrize("extension,encoding,body", [
    ("csv", "utf-8-sig", '姓名,说明\n孙悟空,"花果山,水帘洞\n大闹天宫"\n'),
    ("csv", "gb18030", "姓名,说明\n关羽,过五关\n"),
    ("tsv", "utf-16", "姓名\t说明\n林冲\t风雪山神庙\n"),
])
def test_delimited_tables_preserve_quoted_fields_and_record_numbers(isolated_root, extension, encoding, body):
    reading = read_bytes(isolated_root, "rows." + extension, body.encode(encoding))
    result = assert_chunks(reading)
    assert result.chunks[-1]["metadata"]["row_end"] == 2
    assert result.chunks[0]["metadata"]["sheet_name"] == "数据"
    if extension == "csv" and encoding == "utf-8-sig":
        assert "花果山,水帘洞\n大闹天宫" in result.text


@pytest.mark.parametrize("name,body,expected", [
    ("notes.markdown", "# 标题\n知识库正文", "知识库正文"),
    ("data.json", '{"人物":"关羽"}', "关羽"),
    ("log.log", "中文日志内容", "中文日志内容"),
    ("page.html", "<h1>人物</h1><script>secret()</script><p>孙悟空</p>", "孙悟空"),
    ("page.htm", "<table><tr><td>姓名</td><td>林冲</td></tr></table>", "林冲"),
    ("notes.text", "普通文本", "普通文本"),
])
def test_additional_text_formats(isolated_root, name, body, expected):
    result = assert_chunks(read_bytes(isolated_root, name, body.encode()))
    assert expected in result.text
    assert "secret()" not in result.text


def test_real_ocr_is_offline_and_reuses_engine(isolated_root, ocr_config, monkeypatch):
    import requests
    def reject_network(*args, **kwargs):
        raise AssertionError("OCR must not access the network")
    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
    reading = read_bytes(isolated_root, "scan.png", image_bytes(), ocr_config)
    result = assert_chunks(reading)
    assert "LOCAL LIBRARY" in result.text.upper()
    assert "ALICE" in result.text.upper()
    meta = result.chunks[0]["metadata"]
    assert meta["page_number"] == 1 and meta["extraction_method"] == "ocr"
    assert 0 <= meta["ocr_confidence"] <= 1
    cached = ocr._engine.cache_info()
    read_bytes(isolated_root, "again.png", image_bytes(), ocr_config)
    assert ocr._engine.cache_info().hits > cached.hits


def test_mixed_pdf_preserves_pages_without_ocr_on_text_layer(isolated_root, ocr_config, monkeypatch):
    image_pdf = BytesIO()
    with picture() as image:
        image.save(image_pdf, format="PDF", resolution=150)
    writer = PdfWriter()
    writer.append(BytesIO(pdf_bytes(["Original text layer"])))
    writer.append(BytesIO(image_pdf.getvalue()))
    writer.add_blank_page(width=612, height=792)
    combined = BytesIO()
    writer.write(combined)
    from local_rag.documents import pdf
    original = pdf.recognize_pdf_page
    calls = []
    def track(path, page, settings):
        calls.append(page)
        return original(path, page, settings)
    monkeypatch.setattr(pdf, "recognize_pdf_page", track)
    reading = read_bytes(isolated_root, "mixed.pdf", combined.getvalue(), ocr_config)
    result = assert_chunks(reading)
    assert calls == [1]
    assert reading.extraction.page_count == 3
    assert [u.page_number for u in reading.extraction.units] == [1, 2]
    assert "Original text layer" in result.text
    assert "LOCAL LIBRARY" in result.text.upper()
    assert result.chunks[-1]["metadata"]["extraction_method"] == "ocr"


def test_tiff_multiple_frames_keep_source_pages(isolated_root, ocr_config):
    result = assert_chunks(read_bytes(isolated_root, "frames.tiff", image_bytes("TIFF", multiple=True), ocr_config))
    assert result.page_count == 2
    assert {c["metadata"]["page_number"] for c in result.chunks} == {1, 2}


@pytest.mark.parametrize("extension,format", [("jpg", "JPEG"), ("jpeg", "JPEG"), ("bmp", "BMP"), ("webp", "WEBP"), ("tif", "TIFF")])
def test_image_formats_are_routed_through_ocr(isolated_root, ocr_config, monkeypatch, extension, format):
    calls = []
    def recognized(image, settings):
        calls.append(image.size)
        return "识别文本" * 300, {"extraction_method": "ocr"}
    monkeypatch.setattr(ocr, "recognize", recognized)
    result = assert_chunks(read_bytes(isolated_root, "image." + extension, image_bytes(format), ocr_config))
    assert calls == [(1000, 300)]
    assert len(result.chunks) > 1


@pytest.mark.parametrize("name,body,code", [
    ("fake.xlsx", b"not an archive", "file_type_mismatch"),
    ("fake.xls", b"not an xls", "file_type_mismatch"),
    ("broken.json", b"{bad", "invalid_document"),
    ("blank.csv", b"\n,\n", "no_extractable_text"),
    ("bad.png", b"not an image", "invalid_image"),
    ("fake.jpg", image_bytes(), "file_type_mismatch"),
], ids=["xlsx", "xls", "json", "csv", "image", "image_mismatch"])
def test_invalid_new_formats_fail_clearly(isolated_root, name, body, code):
    with pytest.raises(DocumentError) as exc:
        read_bytes(isolated_root, name, body)
    assert exc.value.code == code


def test_ocr_missing_models_disabled_and_limits(isolated_root, ocr_config):
    cases = [
        (replace(ocr_config, ocr_model_dir=isolated_root / "missing"), "ocr_models_unavailable"),
        (replace(ocr_config, ocr_mode="off"), "ocr_disabled"),
        (replace(ocr_config, ocr_max_pixels=500), "image_too_large"),
    ]
    for settings, code in cases:
        with pytest.raises(DocumentError) as exc:
            read_bytes(isolated_root, "scan.png", image_bytes(), settings)
        assert exc.value.code == code
    with pytest.raises(DocumentError) as exc:
        read_bytes(isolated_root, "two.tiff", image_bytes("TIFF", multiple=True), replace(ocr_config, ocr_max_pages=1))
    assert exc.value.code == "too_many_ocr_pages"


@pytest.mark.parametrize("changes,code", [
    ({"spreadsheet_max_cells": 5}, "spreadsheet_too_large"),
    ({"spreadsheet_max_sheets": 1}, "too_many_sheets"),
    ({"max_text_characters": 80}, "extracted_text_too_large"),
])
def test_table_limits(isolated_root, changes, code):
    with pytest.raises(DocumentError) as exc:
        read_bytes(isolated_root, "large.xlsx", workbook_bytes(), DocumentSettings(**changes))
    assert exc.value.code == code


def test_upload_search_restart_delete_new_formats(isolated_root, ocr_config):
    from local_rag.api.app import create_app
    settings = replace(ocr_config, storage_dir=isolated_root / "documents")
    vectors = VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3")
    def app():
        return create_app(settings, vectors, embedder=FakeEmbeddings(), runtime_settings=RuntimeSettings(warmup_on_startup=False))
    ids = []
    with TestClient(app()) as client:
        health = client.get("/health").json()["data"]
        assert {"xlsx", "xls", "png", "csv"} <= set(health["supported_types"])
        assert health["ocr"]["models_present"]
        for name, body, mime, expected_key in [
            ("table.xlsx", workbook_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "sheet_name"),
            ("scan.png", image_bytes(), "image/png", "ocr_engine"),
        ]:
            response = client.post("/documents", files={"file": (name, body, mime)})
            assert response.status_code == 201, response.text
            data = response.json()["data"]
            assert data["status"] == "indexed"
            assert expected_key in data["chunks"][0]["metadata"]
            assert client.get(data["original_url"]).content == body
            ids.append(data["document_id"])
    with TestClient(app()) as client:
        for identifier in ids:
            results = client.post("/search", json={"query": "LIBRARY 孙悟空", "document_id": identifier}).json()["data"]["matches"]
            assert results and all(c["metadata"]["document_id"] == identifier for c in results)
            response = client.delete("/documents/" + identifier)
            assert response.status_code == 200
            assert response.json()["data"]["remaining_vectors"] == 0

def test_cached_formula_uses_saved_value(isolated_root):
    from zipfile import ZipFile
    source = BytesIO(workbook_bytes())
    target = BytesIO()
    with ZipFile(source) as archive, ZipFile(target, "w") as output:
        for name in archive.namelist():
            data = archive.read(name)
            if name == "xl/worksheets/sheet1.xml":
                assert b"<f>1+2</f><v></v>" in data
                data = data.replace(b"<f>1+2</f><v></v>", b"<f>1+2</f><v>3</v>")
            output.writestr(name, data)
    reading = read_bytes(isolated_root, "cached.xlsx", target.getvalue())
    assert "C5" in reading.text and "未计算公式" not in reading.text
    assert not any("没有缓存" in w for w in reading.extraction.warnings)


def test_pdf_force_ocr_and_page_limit(isolated_root, monkeypatch):
    from local_rag.documents import pdf
    calls = []
    def recognize_page(path, index, settings):
        calls.append(index)
        return "OCR page " + str(index), {"extraction_method": "ocr"}
    monkeypatch.setattr(pdf, "recognize_pdf_page", recognize_page)
    reading = read_bytes(isolated_root, "forced.pdf", pdf_bytes(["TEXT LAYER"]), DocumentSettings(ocr_mode="always"))
    assert reading.text == "OCR page 0"
    assert calls == [0]
    with pytest.raises(DocumentError) as exc:
        read_bytes(isolated_root, "large.pdf", pdf_bytes(["one", "two"]),
                   DocumentSettings(ocr_mode="always", ocr_max_pages=1))
    assert exc.value.code == "too_many_ocr_pages"


def test_empty_image_is_not_success_and_missing_model_upload_rolls_back(isolated_root):
    image = Image.new("RGB", (100, 100), "white")
    body = BytesIO()
    image.save(body, format="PNG")
    image.close()
    with pytest.raises(DocumentError) as exc:
        read_bytes(isolated_root, "blank.png", body.getvalue())
    assert exc.value.code == "no_extractable_text"
    from local_rag.api.app import create_app
    settings = DocumentSettings(storage_dir=isolated_root / "documents", ocr_model_dir=isolated_root / "missing")
    vectors = VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3", auto_index=False)
    app = create_app(settings, vectors, embedder=FakeEmbeddings(), runtime_settings=RuntimeSettings(warmup_on_startup=False))
    with TestClient(app) as client:
        response = client.post("/documents", files={"file": ("scan.png", image_bytes(), "image/png")})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "ocr_models_unavailable"
        assert client.get("/documents").json()["data"]["count"] == 0
        assert not list(settings.storage_dir.iterdir())


def test_expanded_and_cropped_table_context_keeps_actual_row_range(isolated_root):
    from local_rag.rag.context import source_span, crop_metadata
    reading = read_bytes(isolated_root, "source.xlsx", workbook_bytes())
    chunked = assert_chunks(reading)
    record = chunked.to_dict()
    anchor = {**record["chunks"][0], "score": .9, "distance": .1}
    unit = reading.to_dict()["units"][0]["metadata"]
    source = source_span(anchor, record, chunked.text,
                         {"section_id": "sheet1", "title": "人物"},
                         unit["start_index"], unit["end_index"], "parent")
    assert source["metadata"]["row_end"] == 5
    row = next(p for p in unit["paragraphs"] if p["row_index"] == 4)
    source["text"] = chunked.text[row["start_index"]:row["end_index"]]
    crop_metadata(source, row["start_index"] - source["metadata"]["start_index"])
    assert source["metadata"]["row_start"] == source["metadata"]["row_end"] == 4
    assert source["metadata"]["cell_ranges"] == ["A4:C4"]
