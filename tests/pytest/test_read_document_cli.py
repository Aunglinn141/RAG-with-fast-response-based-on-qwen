"""Exercise the local reader command through fresh processes without services."""

import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import UUID, uuid4

import pytest

from local_rag.core.config import PROJECT_ROOT
from tests.fixtures.documents import pdf_bytes


def run_cli(root, *arguments):
    child_environment = {
        **os.environ,
        "DOCUMENT_STORAGE_DIR": str(root / "must-not-create-documents"),
        "CHROMA_STORAGE_DIR": str(root / "must-not-create-chroma"),
        "DOCUMENT_REGISTRY_PATH": str(root / "must-not-create-registry.sqlite3"),
        "LOCAL_OLLAMA_URL": "http://127.0.0.1:1",
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-m", "local_rag.cli.read_document", *(str(arg) for arg in arguments)],
        cwd=PROJECT_ROOT, env=child_environment, capture_output=True, timeout=30,
    )
    # UTF-8 is required even with stdout redirected to a pipe on Windows.
    payload = json.loads(result.stdout.decode("utf-8"))
    assert set(payload) == {"success", "data", "error"}
    assert not (root / "must-not-create-documents").exists()
    assert not (root / "must-not-create-chroma").exists()
    assert not (root / "must-not-create-registry.sqlite3").exists()
    return result, payload


def test_cli_reads_chinese_txt_with_given_id_and_utf8_report(isolated_root):
    source = isolated_root / "中文样例.TXT"
    original = " 第一段\t资料。\r\n\r\n第二段内容。".encode("utf-8")
    source.write_bytes(original)
    output = isolated_root / "reports" / "读取结果.json"
    document_id = str(uuid4())
    process, payload = run_cli(isolated_root, source, "--document-id", document_id, "--output", output)
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert payload["success"] is True and payload["error"] is None
    data = payload["data"]
    assert data["document_id"] == document_id
    assert data["filename"] == source.name
    assert data["text"] == "第一段 资料。\n\n第二段内容。"
    assert data["character_count"] == len(data["text"])
    assert data["units"][0]["metadata"]["document_id"] == document_id
    assert data["units"][0]["metadata"]["page_number"] is None
    assert output.read_bytes() == process.stdout.replace(b"\r\n", b"\n")
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert source.read_bytes() == original


def test_cli_reads_pdf_without_output_file_and_preserves_physical_pages(isolated_root):
    source = isolated_root / "sample.pdf"
    original = pdf_bytes(["First physical page", "", "Third physical page"])
    source.write_bytes(original)
    process, payload = run_cli(isolated_root, source)
    assert process.returncode == 0, process.stderr.decode("utf-8")
    data = payload["data"]
    assert UUID(data["document_id"])
    assert data["page_count"] == 3
    assert [unit["metadata"]["page_number"] for unit in data["units"]] == [1, 3]
    assert data["warnings"]
    assert sorted(path.name for path in isolated_root.iterdir()) == [source.name]
    assert source.read_bytes() == original


@pytest.mark.parametrize("name,body,error", [
    ("unsupported.exe", b"not supported", "unsupported_file_type"),
    ("empty.txt", b"", "empty_file"),
    ("blank.txt", b" \r\n \t ", "no_extractable_text"),
    ("no-text.pdf", pdf_bytes([""]), "no_extractable_text"),
    ("missing.txt", None, "document_file_not_found"),
])
def test_cli_returns_error_json_without_creating_report(isolated_root, name, body, error):
    source = isolated_root / name
    if body is not None:
        source.write_bytes(body)
    output = isolated_root / "result.json"
    process, payload = run_cli(isolated_root, source, "--output", output)
    assert process.returncode == 1
    assert payload["success"] is False and payload["data"] is None
    assert payload["error"]["code"] == error
    assert payload["error"]["message"]
    if name.endswith(".pdf"):
        assert "OCR" in payload["error"]["message"]
    assert not output.exists()
    if body is not None:
        assert source.read_bytes() == body


def test_cli_rejects_invalid_document_uuid(isolated_root):
    source = isolated_root / "sample.txt"
    source.write_text("文本", encoding="utf-8")
    process, payload = run_cli(isolated_root, source, "--document-id", "not-a-uuid")
    assert process.returncode == 1
    assert payload["error"]["code"] == "invalid_document_id"


def test_cli_does_not_overwrite_existing_output(isolated_root):
    source, output = isolated_root / "sample.txt", isolated_root / "existing.json"
    source.write_text("中文原文", encoding="utf-8")
    original = b'{"keep":"existing report"}'
    output.write_bytes(original)
    process, payload = run_cli(isolated_root, source, "--output", output)
    assert process.returncode == 1
    assert payload["error"]["code"] == "output_exists"
    assert output.read_bytes() == original


def test_cli_rejects_source_as_output_even_when_extension_is_not_json(isolated_root):
    source = isolated_root / "sample.txt"
    original = "保护原文。".encode("utf-8")
    source.write_bytes(original)
    process, payload = run_cli(isolated_root, source, "--output", source)
    assert process.returncode == 1
    assert payload["error"]["code"] == "output_is_source"
    assert source.read_bytes() == original


def test_cli_rejects_hardlink_to_source_as_output(isolated_root):
    source, alias = isolated_root / "sample.txt", isolated_root / "alias.json"
    original = "同一文件的别名。".encode("utf-8")
    source.write_bytes(original)
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"This filesystem cannot create hard links: {exc}")
    process, payload = run_cli(isolated_root, source, "--output", alias)
    assert process.returncode == 1
    assert payload["error"]["code"] == "output_is_source"
    assert source.read_bytes() == alias.read_bytes() == original


def test_cli_requires_json_output_extension(isolated_root):
    source, output = isolated_root / "sample.txt", isolated_root / "result.csv"
    source.write_text("中文原文", encoding="utf-8")
    process, payload = run_cli(isolated_root, source, "--output", output)
    assert process.returncode == 1
    assert payload["error"]["code"] == "invalid_output_path"
    assert not output.exists()
