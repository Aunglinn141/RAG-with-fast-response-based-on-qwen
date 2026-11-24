"""Exercise real uploads, parsers, storage and LangChain splitting end to end."""

import shutil
import unittest
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from local_rag.api.app import create_app as build_app
from local_rag.core.config import DocumentSettings, VectorSettings, RUNTIME_DIR
from tests.fixtures.documents import docx_bytes, pdf_bytes


def create_app(settings):
    # Step-2 parser tests deliberately exercise extraction without a model dependency.
    return build_app(settings, VectorSettings(
        chroma_dir=settings.storage_dir.parent / "chroma",
        registry_path=settings.storage_dir.parent / "registry.sqlite3", auto_index=False))


class DocumentAPITests(unittest.TestCase):
    def setUp(self):
        scratch = RUNTIME_DIR / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.test_root = scratch / str(uuid4())
        self.storage = self.test_root / "documents"
        self.storage.mkdir(parents=True)
        self.settings = DocumentSettings(storage_dir=self.storage)
        self.client = TestClient(create_app(self.settings))

    def tearDown(self):
        self.client.close()
        expected_parent = RUNTIME_DIR / "tests"
        if self.test_root.resolve().parent == expected_parent.resolve():
            shutil.rmtree(self.test_root)

    def upload(self, filename, data, content_type="application/octet-stream", client=None):
        return (client or self.client).post("/documents", files={"file": (filename, data, content_type)})

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertIs(response.json()["success"], False)
        self.assertIsNone(response.json()["data"])
        self.assertEqual(response.json()["error"]["code"], code)
        self.assertTrue(response.json()["error"]["message"])

    def assert_source_mapping(self, record, original):
        UUID(record["document_id"])
        self.assertEqual(record["status"], "ready")
        self.assertEqual(self.client.get(record["original_url"]).content, original)
        text = self.client.get(record["text_url"]).content.decode("utf-8")
        self.assertEqual(record["character_count"], len(text))
        self.assertEqual(record["chunk_count"], len(record["chunks"]))
        for index, chunk in enumerate(record["chunks"], 1):
            metadata = chunk["metadata"]
            self.assertEqual(metadata["document_id"], record["document_id"])
            self.assertEqual(metadata["filename"], record["filename"])
            self.assertEqual(metadata["chunk_index"], index)
            self.assertEqual(chunk["chunk_id"], f"{record['document_id']}:{index}")
            self.assertGreater(len(chunk["text"]), 0)
            self.assertLessEqual(len(chunk["text"]), record["chunk_config"]["chunk_size"])
            self.assertEqual(text[metadata["start_index"]:metadata["end_index"]], chunk["text"])
        return text

    def test_txt_upload_storage_whitespace_and_unique_ids(self):
        original = ("  第一段\t  内容。\r\n\r\n\r\n第二段：" + "文本分块验证。" * 220).encode("utf-8")
        response = self.upload("示例.TXT", original, "text/plain")
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()["data"]
        text = self.assert_source_mapping(record, original)
        self.assertTrue(text.startswith("第一段 内容。\n\n第二段："))
        self.assertNotIn("\r", text)
        self.assertGreater(record["chunk_count"], 1)
        self.assertTrue(all(c["metadata"]["page_number"] is None for c in record["chunks"]))
        second = self.upload("示例.TXT", original).json()["data"]
        self.assertNotEqual(record["document_id"], second["document_id"])
        self.assertEqual(record["sha256"], second["sha256"])

    def test_exact_character_fallback_overlap(self):
        # No separators: LangChain must fall back to 700 characters with 100 overlap.
        original = ("甲乙丙丁戊己庚辛壬癸" * 150).encode("utf-8")
        record = self.upload("overlap.txt", original).json()["data"]
        self.assert_source_mapping(record, original)
        chunks = record["chunks"]
        self.assertEqual([len(c["text"]) for c in chunks], [700, 700, 300])
        self.assertEqual(chunks[1]["metadata"]["start_index"], 600)
        self.assertEqual(chunks[0]["text"][-100:], chunks[1]["text"][:100])

    def test_pdf_page_numbers_and_chunks_do_not_cross_pages(self):
        original = pdf_bytes(["PageOne " * 160, "PageTwo " * 150])
        response = self.upload("pages.pdf", original, "application/pdf")
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()["data"]
        self.assert_source_mapping(record, original)
        self.assertEqual(record["page_count"], 2)
        self.assertEqual({c["metadata"]["page_number"] for c in record["chunks"]}, {1, 2})
        for chunk in record["chunks"]:
            self.assertNotIn("PageTwo" if chunk["metadata"]["page_number"] == 1 else "PageOne", chunk["text"])

    def test_blank_pdf_page_keeps_physical_number_and_warns(self):
        record = self.upload("blank-first.pdf", pdf_bytes(["", "Second physical page"])).json()["data"]
        self.assertEqual(record["chunks"][0]["metadata"]["page_number"], 2)
        self.assertEqual(record["page_count"], 2)
        self.assertIn("第 1 页", record["warnings"][0])

    def test_docx_preserves_headings_paragraphs_and_tables(self):
        original = docx_bytes()
        response = self.upload("guide.docx", original)
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()["data"]
        text = self.assert_source_mapping(record, original)
        self.assertIn("Table key\n\nTable value", text)
        first = [c for c in record["chunks"] if c["metadata"]["section_title"] == "First section"]
        self.assertGreater(len(first), 1)
        self.assertEqual(first[0]["metadata"]["heading_level"], 1)
        for chunk in record["chunks"]:
            metadata = chunk["metadata"]
            self.assertIsNone(metadata["page_number"])
            self.assertTrue(metadata["paragraphs"])
            self.assertGreaterEqual(metadata["paragraph_start"], 1)
            self.assertGreaterEqual(metadata["paragraph_end"], metadata["paragraph_start"])
        self.assertEqual(record["chunks"][-1]["metadata"]["section_title"], "Second section")

    def test_markdown_becomes_plain_text_with_heading_metadata(self):
        original = "# 概览\n\n这是 **重点**，参考 [说明](https://example.test)。\n\n## 示例\n\n```python\nprint('local')\n```\n".encode()
        record = self.upload("guide.md", original, "text/markdown").json()["data"]
        text = self.assert_source_mapping(record, original)
        self.assertIn("这是 重点，参考 说明。", text)
        self.assertIn("print('local')", text)
        self.assertNotIn("```", text)
        self.assertNotIn("https://", text)
        self.assertEqual(record["chunks"][0]["metadata"]["section_title"], "概览")
        self.assertEqual(record["chunks"][-1]["metadata"]["heading_level"], 2)

    def test_supported_chinese_encodings(self):
        for encoding in ("utf-8-sig", "utf-16", "utf-32", "gb18030"):
            with self.subTest(encoding=encoding):
                record = self.upload("编码.txt", "中文编码测试。".encode(encoding)).json()["data"]
                self.assertEqual(record["chunks"][0]["text"], "中文编码测试。")

    def test_document_survives_app_restart(self):
        record = self.upload("persist.txt", b"Saved locally.").json()["data"]
        with TestClient(create_app(self.settings)) as restarted:
            response = restarted.get(f"/documents/{record['document_id']}")
            self.assertEqual(response.json()["data"], record)
            self.assertEqual(restarted.get(record["chunks_url"]).json()["data"]["chunks"], record["chunks"])

    def test_unsupported_files_and_mime_mismatch(self):
        for filename in ("program.exe", "old.doc", "sheet.xlsb", "noextension"):
            with self.subTest(filename=filename):
                self.assert_error(self.upload(filename, b"content"), 415, "unsupported_file_type")
        self.assert_error(self.upload("fake.pdf", b"not a PDF"), 415, "file_type_mismatch")
        self.assert_error(self.upload("wrong.txt", b"content", "application/pdf"), 415, "file_type_mismatch")
        self.assert_error(self.upload("renamed.txt", pdf_bytes(["Test"])), 415, "file_type_mismatch")
        self.assertFalse(list(self.storage.iterdir()))

    def test_empty_binary_and_unreadable_documents(self):
        cases = [
            ("empty.txt", b"", "empty_file"),
            ("blank.txt", b" \n\t\n ", "no_extractable_text"),
            ("binary.txt", b"\x00\x01binary", "binary_text_file"),
            ("broken.pdf", b"%PDF-1.7\nnot a valid PDF", "invalid_document"),
            ("broken.docx", b"not a zip", "invalid_document"),
            ("scanned.pdf", pdf_bytes([""]), "no_extractable_text"),
            ("locked.pdf", pdf_bytes(["Private"], password="secret"), "encrypted_pdf"),
        ]
        for filename, data, code in cases:
            with self.subTest(filename=filename):
                self.assert_error(self.upload(filename, data), 415 if code == "binary_text_file" else 422, code)
                self.assertFalse(list(self.storage.iterdir()))

    def test_file_size_limit_and_exact_boundary(self):
        settings = DocumentSettings(storage_dir=self.storage, max_file_bytes=256)
        with TestClient(create_app(settings)) as client:
            self.assertEqual(self.upload("boundary.txt", b"x" * 256, client=client).status_code, 201)
            self.assert_error(self.upload("oversize.txt", b"x" * 257, client=client), 413, "file_too_large")
        self.assertFalse(list(self.storage.glob("*.pending")))

    def test_upload_larger_than_multipart_memory_spool(self):
        original = b"Local document processing.\n" * 42000
        self.assertGreater(len(original), 1024 * 1024)
        response = self.upload("large.txt", original)
        self.assertEqual(response.status_code, 201, response.text[:300])
        self.assert_source_mapping(response.json()["data"], original)

    def test_streaming_size_limit_without_honest_content_length(self):
        settings = DocumentSettings(storage_dir=self.storage, max_file_bytes=256)
        body = b'--X\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\n\r\n' + b"a" * 70000 + b"\r\n--X--\r\n"
        with TestClient(create_app(settings)) as client:
            for headers in ({"content-length": "1"}, {}):
                with self.subTest(headers=headers):
                    response = client.post("/documents", content=iter([body[:100], body[100:]]), headers={
                        "content-type": "multipart/form-data; boundary=X", **headers,
                    })
                    self.assert_error(response, 413, "file_too_large")
        self.assertFalse(list(self.storage.iterdir()))

    def test_invalid_upload_form_and_unknown_document(self):
        self.assert_error(self.client.post("/documents", json={}), 400, "invalid_upload")
        self.assert_error(self.client.post("/documents", files={"wrong": ("a.txt", b"text")}), 400, "invalid_upload")
        self.assert_error(self.client.post("/documents", files=[("file", ("a.txt", b"a")), ("file", ("b.txt", b"b"))]), 400, "invalid_upload")
        self.assert_error(self.client.get(f"/documents/{uuid4()}"), 404, "document_not_found")
        self.assert_error(self.client.get("/documents/not-a-uuid"), 404, "document_not_found")

    def test_filename_paths_do_not_escape_storage(self):
        for filename in ("../../safe.txt", "C:\\fakepath\\safe.txt"):
            with self.subTest(filename=filename):
                response = self.upload(filename, b"Safe original")
                self.assertEqual(response.status_code, 201, response.text)
                record = response.json()["data"]
                self.assertEqual(record["filename"], "safe.txt")
                self.assertTrue((self.storage / record["document_id"] / "original.txt").is_file())

    def test_chunk_settings_can_vary_by_document_type(self):
        settings = DocumentSettings(storage_dir=self.storage, chunk_overrides={"txt": (100, 20)})
        with TestClient(create_app(settings)) as client:
            record = self.upload("custom.txt", b"x" * 250, client=client).json()["data"]
            self.assertEqual(record["chunk_config"], {"chunk_size": 100, "chunk_overlap": 20})
            self.assertEqual([len(c["text"]) for c in record["chunks"]], [100, 100, 90])

    def test_health_and_upload_schema(self):
        self.assertEqual(self.client.get("/health").json()["data"]["chunk_size"], 700)
        schema = self.client.get("/openapi.json").json()
        body = schema["paths"]["/documents"]["post"]["requestBody"]
        self.assertEqual(body["content"]["multipart/form-data"]["schema"]["properties"]["file"]["format"], "binary")


if __name__ == "__main__":
    unittest.main()
