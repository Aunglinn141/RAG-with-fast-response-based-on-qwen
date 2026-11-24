"""Atomic local cache for derived hierarchy; original documents remain authoritative."""

import json
from uuid import uuid4

from local_rag.documents.hierarchy import VERSION, build_hierarchy, fingerprint


def load_hierarchy(directory, record, parent_characters=2100):
    text = (directory / "text.txt").read_text(encoding="utf-8")
    expected = fingerprint(record, text, parent_characters)
    path = directory / "retrieval.json"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (cached["version"] == VERSION and cached["fingerprint"] == expected
                and cached["document_id"] == record["document_id"]
                and set(cached["children"]) == {c["chunk_id"] for c in record["chunks"]}
                and all(0 <= p["start_index"] < p["end_index"] <= len(text)
                        for p in [*cached["parents"], *cached["sections"]])):
            return text, cached
    except (OSError, ValueError, KeyError, TypeError):
        pass
    hierarchy = build_hierarchy(record, text, parent_characters)
    temporary = directory / ("retrieval." + uuid4().hex + ".pending")
    try:
        temporary.write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return text, hierarchy
