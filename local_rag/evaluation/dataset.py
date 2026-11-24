"""Load and validate a versioned corpus, gold answers and exact supporting text."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from local_rag.core.config import PROJECT_ROOT

DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "chinese_rag_v1" / "dataset.json"


@dataclass(frozen=True)
class BenchmarkDataset:
    manifest: dict
    documents: dict[str, bytes]
    sha256: str
    file_sha256: dict[str, str]

    @property
    def cases(self):
        return self.manifest["cases"]


def load_dataset(path=DEFAULT_DATASET):
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    files = manifest.get("corpus_files", [])
    if not files or len(set(files)) != len(files):
        raise ValueError("Corpus filenames must be nonempty and unique")
    documents, file_hashes = {}, {"dataset.json": hashlib.sha256(raw).hexdigest()}
    digest = hashlib.sha256()
    # Length prefixes avoid ambiguity between filename/content boundaries.
    def include(name, content):
        for value in (name.encode("utf-8"), content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    include("dataset.json", raw)
    for filename in sorted(files):
        if Path(filename).name != filename or Path(filename).suffix not in {".md", ".txt"}:
            raise ValueError("Corpus files must be TXT/MD basenames")
        content = (path.parent / "corpus" / filename).read_bytes()
        if not content.decode("utf-8").strip():
            raise ValueError(f"Empty corpus document: {filename}")
        documents[filename] = content
        file_hashes[filename] = hashlib.sha256(content).hexdigest()
        include(filename, content)
    cases, seen = manifest.get("cases", []), set()
    if not cases:
        raise ValueError("The benchmark has no questions")
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Case IDs must be nonempty and unique")
        seen.add(case_id)
        if not case.get("question") or not case.get("reference_answer"):
            raise ValueError(f"Question and reference answer required: {case_id}")
        if type(case.get("should_abstain")) is not bool:
            raise ValueError(f"Boolean should_abstain required: {case_id}")
        expected = case.get("expected_source_files", [])
        groups = case.get("required_fact_groups", [])
        quotes = case.get("support_quotes", [])
        if len(set(expected)) != len(expected) or any(name not in documents for name in expected):
            raise ValueError(f"Unknown or duplicate gold source: {case_id}")
        if any(not group or any(not isinstance(alias, str) or not alias.strip() for alias in group)
               for group in groups):
            raise ValueError(f"Fact groups must contain nonempty text aliases: {case_id}")
        if case["should_abstain"] and (expected or groups or quotes):
            raise ValueError(f"Unanswerable cases must not have positive gold facts: {case_id}")
        if not case["should_abstain"] and (not expected or not groups or not quotes):
            raise ValueError(f"Answerable cases need sources, fact groups and quotations: {case_id}")
        for quote in quotes:
            if (quote.get("file") not in expected or not quote.get("text")
                    or quote["text"] not in documents[quote["file"]].decode("utf-8")):
                raise ValueError(f"Gold quote is not verbatim in its document: {case_id}")
        if not case["should_abstain"] and set(expected) != {quote["file"] for quote in quotes}:
            raise ValueError(f"Every expected source requires a gold quote: {case_id}")
        if case.get("document_scope") and case["document_scope"] not in documents:
            raise ValueError(f"Unknown document scope: {case_id}")
        if case.get("document_scope") and any(name != case["document_scope"] for name in expected):
            raise ValueError(f"Gold answer exceeds document scope: {case_id}")
    fingerprint = digest.hexdigest()
    checksum = path.parent / "SHA256"
    if checksum.exists() and checksum.read_text(encoding="ascii").strip() != fingerprint:
        raise ValueError("Frozen dataset fingerprint differs from SHA256; use a new version for changes")
    return BenchmarkDataset(manifest, documents, fingerprint, file_hashes)
