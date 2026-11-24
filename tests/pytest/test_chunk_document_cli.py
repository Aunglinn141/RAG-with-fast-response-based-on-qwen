"""Exercise standalone chunking through real subprocesses and local documents."""

import json
import os
import subprocess
import sys
from uuid import UUID, uuid4

import pytest

from local_rag.core.config import PROJECT_ROOT
from tests.fixtures.documents import pdf_bytes


def run_cli(root, *arguments, extra_env=None, parse_json=True):
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("DOCUMENT_")
    }
    environment.update({
        "DOCUMENT_STORAGE_DIR": str(root / "must-not-create-documents"),
        "CHROMA_STORAGE_DIR": str(root / "must-not-create-chroma"),
        "DOCUMENT_REGISTRY_PATH": str(root / "must-not-create-registry.sqlite3"),
        "LOCAL_OLLAMA_URL": "http://127.0.0.1:1",
        "PYTHONIOENCODING": "utf-8",
    })
    environment.update(extra_env or {})
    process = subprocess.run(
        [sys.executable, "-m", "local_rag.cli.chunk_document", *(str(arg) for arg in arguments)],
        cwd=PROJECT_ROOT, env=environment, capture_output=True, timeout=30,
    )
    assert not (root / "must-not-create-documents").exists()
    assert not (root / "must-not-create-chroma").exists()
    assert not (root / "must-not-create-registry.sqlite3").exists()
    payload = json.loads(process.stdout.decode("utf-8")) if parse_json else None
    if parse_json:
        assert set(payload) == {"success", "data", "error"}
    return process, payload


def source_file(root):
    source = root / "中文分块样例.TXT"
    source.write_text("".join(chr(0x4E00 + index) for index in range(1800)), encoding="utf-8")
    return source


def assert_mapping(data, filename, document_id=None):
    assert str(UUID(data["document_id"])) == data["document_id"]
    if document_id is not None:
        assert data["document_id"] == document_id
    assert data["filename"] == filename
    assert data["chunk_count"] == len(data["chunks"])
    assert data["character_count"] == len(data["text"])
    for index, chunk in enumerate(data["chunks"], 1):
        metadata = chunk["metadata"]
        assert metadata["document_id"] == data["document_id"]
        assert metadata["filename"] == filename
        assert metadata["chunk_index"] == index
        assert chunk["chunk_id"] == f"{data['document_id']}:{index}"
        assert "page_number" in metadata
        assert data["text"][metadata["start_index"]:metadata["end_index"]] == chunk["text"]
        assert 0 < len(chunk["text"]) <= data["chunk_config"]["chunk_size"]


@pytest.mark.parametrize("arguments,size,overlap,lengths,starts", [
    ([], 700, 100, [700, 700, 600], [0, 600, 1200]),
    (["--chunk-size", "500", "--chunk-overlap", "80"], 500, 80,
     [500, 500, 500, 500, 120], [0, 420, 840, 1260, 1680]),
    (["--chunk-size", "800", "--chunk-overlap", "120"], 800, 120,
     [800, 800, 440], [0, 680, 1360]),
])
def test_cli_character_limits_overlap_metadata_and_utf8_export(
        isolated_root, arguments, size, overlap, lengths, starts):
    source = source_file(isolated_root)
    original = source.read_bytes()
    document_id = str(uuid4())
    output = isolated_root / "reports" / "分块结果.json"
    process, payload = run_cli(isolated_root, source, *arguments,
                               "--document-id", document_id, "--output", output)
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert payload["success"] is True and payload["error"] is None
    data = payload["data"]
    assert_mapping(data, source.name, document_id)
    assert data["chunk_config"] == {"chunk_size": size, "chunk_overlap": overlap}
    assert [len(chunk["text"]) for chunk in data["chunks"]] == lengths
    assert [chunk["metadata"]["start_index"] for chunk in data["chunks"]] == starts
    assert all(chunk["metadata"]["page_number"] is None for chunk in data["chunks"])
    for previous, current in zip(data["chunks"], data["chunks"][1:]):
        assert previous["text"][-overlap:] == current["text"][:overlap]
    assert output.read_bytes() == process.stdout.replace(b"\r\n", b"\n")
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert source.read_bytes() == original


@pytest.mark.parametrize("environment,arguments,expected", [
    ({"DOCUMENT_CHUNK_SIZE": "500", "DOCUMENT_CHUNK_OVERLAP": "80"}, [], (500, 80)),
    ({"DOCUMENT_CHUNK_SIZE": "not-an-integer", "DOCUMENT_CHUNK_OVERLAP": "bad"},
     ["--chunk-size", "800", "--chunk-overlap", "120"], (800, 120)),
    ({"DOCUMENT_CHUNK_SIZE": "600", "DOCUMENT_CHUNK_OVERLAP": "90"}, ["--chunk-size", "700"], (700, 90)),
    ({"DOCUMENT_CHUNK_SIZE": "600", "DOCUMENT_CHUNK_OVERLAP": "bad"}, ["--chunk-overlap", "0"], (600, 0)),
])
def test_cli_parameters_override_only_corresponding_environment(isolated_root, environment, arguments, expected):
    source = source_file(isolated_root)
    process, payload = run_cli(isolated_root, source, *arguments, extra_env=environment)
    assert process.returncode == 0, process.stderr.decode("utf-8")
    assert payload["data"]["chunk_config"] == dict(zip(("chunk_size", "chunk_overlap"), expected))
    assert_mapping(payload["data"], source.name)
    assert list(isolated_root.iterdir()) == [source]


@pytest.mark.parametrize("arguments,environment", [
    (["--chunk-size", "49"], {}),
    (["--chunk-size", "10001"], {}),
    (["--chunk-overlap", "-1"], {}),
    (["--chunk-size", "500", "--chunk-overlap", "500"], {}),
    ([], {"DOCUMENT_CHUNK_SIZE": "bad"}),
])
def test_invalid_configuration_returns_error_json_without_output(isolated_root, arguments, environment):
    source = source_file(isolated_root)
    output = isolated_root / "should-not-exist.json"
    process, payload = run_cli(isolated_root, source, *arguments, "--output", output, extra_env=environment)
    assert process.returncode == 1
    assert payload["success"] is False and payload["data"] is None
    assert payload["error"]["code"] == "invalid_configuration"
    assert not output.exists()


def test_cli_non_integer_argument_is_usage_error(isolated_root):
    source = source_file(isolated_root)
    process, _ = run_cli(isolated_root, source, "--chunk-size", "700.5", parse_json=False)
    assert process.returncode == 2
    assert b"--chunk-size" in process.stderr
    assert not process.stdout


def test_cli_pdf_preserves_page_number_and_boundaries(isolated_root):
    source = isolated_root / "pages.pdf"
    source.write_bytes(pdf_bytes(["FirstPage " * 100, "", "ThirdPage " * 100]))
    process, payload = run_cli(isolated_root, source)
    assert process.returncode == 0, process.stderr.decode("utf-8")
    data = payload["data"]
    assert_mapping(data, source.name)
    assert data["page_count"] == 3 and data["warnings"]
    assert {chunk["metadata"]["page_number"] for chunk in data["chunks"]} == {1, 3}
    for chunk in data["chunks"]:
        assert ("FirstPage" in chunk["text"]) != ("ThirdPage" in chunk["text"])


@pytest.mark.parametrize("name,body,expected_code", [
    ("empty.txt", b"", "empty_file"),
    ("unsupported.exe", b"text", "unsupported_file_type"),
])
def test_invalid_documents_return_error_and_no_report(isolated_root, name, body, expected_code):
    source = isolated_root / name
    source.write_bytes(body)
    output = isolated_root / "result.json"
    process, payload = run_cli(isolated_root, source, "--output", output)
    assert process.returncode == 1
    assert payload["error"]["code"] == expected_code
    assert payload["data"] is None and not output.exists()
    assert source.read_bytes() == body


@pytest.mark.parametrize("destination,expected_code", [
    ("existing", "output_exists"), ("source", "output_is_source"), ("csv", "invalid_output_path"),
])
def test_output_preflight_protects_source_and_existing_reports(isolated_root, destination, expected_code):
    source = source_file(isolated_root)
    original = source.read_bytes()
    output = isolated_root / "result.json"
    old_report = b'{"preserve":"original report"}'
    if destination == "existing":
        output.write_bytes(old_report)
    elif destination == "source":
        output = source
    else:
        output = isolated_root / "result.csv"
    process, payload = run_cli(isolated_root, source, "--output", output)
    assert process.returncode == 1
    assert payload["error"]["code"] == expected_code
    assert source.read_bytes() == original
    if destination == "existing":
        assert output.read_bytes() == old_report
    elif destination == "csv":
        assert not output.exists()


def test_output_hardlink_cannot_replace_source(isolated_root):
    source = source_file(isolated_root)
    original = source.read_bytes()
    output = isolated_root / "alias.json"
    try:
        os.link(source, output)
    except OSError as exc:
        pytest.skip(f"This filesystem cannot create hard links: {exc}")
    process, payload = run_cli(isolated_root, source, "--output", output)
    assert process.returncode == 1
    assert payload["error"]["code"] == "output_is_source"
    assert source.read_bytes() == output.read_bytes() == original
