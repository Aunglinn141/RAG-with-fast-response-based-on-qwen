"""Resolve semantic child hits into bounded original context and summary evidence."""

from copy import deepcopy

from local_rag.documents.hierarchy import extract_summary, terms
from local_rag.rag.retrieval import select_context_candidates


def source_span(anchor, record, text, section, start, end, kind, *, parent_id=None, focus=None):
    """Every returned text is one contiguous slice, including summary sentences."""
    overlapping = [c for c in record["chunks"]
                   if c["metadata"]["start_index"] < end and c["metadata"]["end_index"] > start]
    if not overlapping or start >= end:
        return None
    metadata = deepcopy(overlapping[0]["metadata"])
    for field in ("retrieval_channels", "score_kind"):
        if field in anchor.get("metadata", {}):
            metadata[field] = anchor["metadata"][field]
    pages = sorted({c["metadata"]["page_number"] for c in overlapping if c["metadata"].get("page_number") is not None})
    paragraphs = {p["paragraph_index"]: p for c in overlapping for p in c["metadata"].get("paragraphs", [])
                  if p["start_index"] < end and p["end_index"] > start}
    metadata.update(
        document_id=record["document_id"], filename=record["filename"],
        start_index=start, end_index=end, page_number=pages[0] if len(pages) == 1 else None,
        page_numbers=pages, section_id=section["section_id"], section_title=section["title"],
        context_type=kind, parent_id=parent_id, anchor_chunk_id=anchor["chunk_id"],
        source_chunk_ids=[c["chunk_id"] for c in overlapping],
        source_chunks=[{"chunk_id": c["chunk_id"], "start_index": c["metadata"]["start_index"],
                        "end_index": c["metadata"]["end_index"], "chunk_index": c["metadata"]["chunk_index"]} for c in overlapping],
        chunk_index=overlapping[0]["metadata"]["chunk_index"],
        chunk_index_end=overlapping[-1]["metadata"]["chunk_index"],
        paragraphs=list(paragraphs.values()),
        paragraph_start=min(paragraphs) if paragraphs else None,
        paragraph_end=max(paragraphs) if paragraphs else None,
        focus_start=focus[0] if focus else start, focus_end=focus[1] if focus else end,
    )
    _refresh_table_location(metadata)
    return {**anchor, "chunk_id": f"{record['document_id']}:context:{start}:{end}",
            "text": text[start:end], "metadata": metadata}


def expand_context(question, matches, documents, plan, limit):
    """Plan on already-visible matches. This function never searches another scope."""
    anchors = select_context_candidates(matches, len(matches))
    windows, chapter_hits = [], {}
    for anchor in anchors:
        key = anchor["metadata"]["document_id"]
        if key not in documents:
            windows.append(anchor)
            continue
        record, text, hierarchy = documents[key]
        chunks = {c["chunk_id"]: c for c in record["chunks"]}
        child = chunks.get(anchor["chunk_id"])
        if not child:
            continue
        lo, hi = child["metadata"]["start_index"], child["metadata"]["end_index"]
        parents = {p["parent_id"]: p for p in hierarchy["parents"]}
        sections = {s["section_id"]: s for s in hierarchy["sections"]}
        possible = [parents[p] for p in hierarchy["children"].get(child["chunk_id"], [])]
        if not possible:
            windows.append(anchor)
            continue
        # Prefer the parent that contains the most of the actual child match.
        parent = max(possible, key=lambda p: min(hi, p["end_index"]) - max(lo, p["start_index"]))
        section = sections[parent["section_id"]]
        for p in possible:
            section_key = (key, p["section_id"])
            if section_key not in chapter_hits:
                chapter_hits[section_key] = anchor
        start = parent["start_index"] if plan.parent_child else max(lo, section["start_index"])
        end = parent["end_index"] if plan.parent_child else min(hi, section["end_index"])
        neighbor_ids = []
        position = next(i for i, c in enumerate(record["chunks"]) if c["chunk_id"] == child["chunk_id"])
        for neighbor in record["chunks"][max(0, position - plan.neighbor_radius):position + plan.neighbor_radius + 1]:
            a, b = neighbor["metadata"]["start_index"], neighbor["metadata"]["end_index"]
            # Do not cross a chapter, source unit, page boundary or document.
            if a >= section["end_index"] or b <= section["start_index"]:
                continue
            a, b = max(a, section["start_index"]), min(b, section["end_index"])
            if a < start or b > end:
                neighbor_ids.append(neighbor["chunk_id"])
            start, end = min(start, a), max(end, b)
        core = extract_summary(text, max(lo, start), min(hi, end), question, max_sentences=1)
        focus = (core[0]["start_index"], core[0]["end_index"]) if core else (max(lo, start), min(hi, end))
        kind = "parent" if plan.parent_child else "neighbor" if neighbor_ids else "child"
        source = source_span(anchor, record, text, section, start, end, kind,
                             parent_id=parent["parent_id"] if plan.parent_child else None, focus=focus)
        if source:
            source["metadata"]["neighbor_chunk_ids"] = neighbor_ids
            windows.append(source)

    summaries = []
    if plan.section_summaries:
        ranked_sections = []
        for (doc_id, section_id), anchor in chapter_hits.items():
            record, text, hierarchy = documents[doc_id]
            section = next(s for s in hierarchy["sections"] if s["section_id"] == section_id)
            overlap = len(terms(question) & terms(section["title"] + section["summary"]))
            ranked_sections.append((anchor["score"] + .01 * min(overlap, 5), doc_id, section, anchor))
        # Preserve relevance, but let different documents contribute before a
        # second chapter from one document, just as in child candidate selection.
        ranked_sections.sort(key=lambda row: -row[0])
        used_docs = set()
        ordered = []
        for row in ranked_sections:
            if row[1] not in used_docs:
                ordered.append(row)
                used_docs.add(row[1])
        ordered.extend(row for row in ranked_sections if row not in ordered)
        for _, doc_id, section, anchor in ordered[:max(1, limit // 2)]:
            record, text, _ = documents[doc_id]
            focused = extract_summary(text, section["start_index"], section["end_index"], question, max_sentences=2)
            for span in focused:
                source = source_span(anchor, record, text, section, span["start_index"], span["end_index"], "summary")
                if source:
                    source["metadata"]["summary_method"] = "extractive-v2"
                    summaries.append(source)
        # Round-robin sentences across chapters, so one chapter cannot use all
        # the summary slots before the next relevant chapter has been seen.
        grouped = {}
        for source in summaries:
            grouped.setdefault(source["metadata"]["section_id"], []).append(source)
        summaries = [group[i] for i in range(2) for group in grouped.values() if len(group) > i]
    # Leave room for original parent windows even for overview questions.
    summary_slots = max(1, limit // 2) if summaries else 0
    candidates = summaries[:summary_slots] + windows + summaries[summary_slots:]
    selected, texts = [], set()
    for source in candidates:
        key = source["text"].strip()
        if key in texts:
            continue
        meta = source["metadata"]
        redundant = any(prior["metadata"].get("document_id") == meta.get("document_id")
               and "start_index" in prior["metadata"]
               and min(meta["end_index"], prior["metadata"]["end_index"]) - max(meta["start_index"], prior["metadata"]["start_index"])
               > .85 * max(meta["end_index"] - meta["start_index"], prior["metadata"]["end_index"] - prior["metadata"]["start_index"])
               for prior in selected) if "start_index" in meta else False
        if redundant:
            continue
        selected.append(source)
        texts.add(key)
        if len(selected) == limit:
            break
    return selected


def fit_source(source, length):
    """Crop around the matching sentence, preserving exact absolute provenance."""
    if length >= len(source["text"]) or "focus_start" not in source["metadata"]:
        return source["text"][:length], 0
    meta = source["metadata"]
    relative = meta["focus_start"] - meta["start_index"]
    focus_length = min(length, max(0, meta["focus_end"] - meta["focus_start"]))
    start = max(0, min(relative - (length - focus_length) // 2, len(source["text"]) - length))
    end = start + length
    # Keep sentence boundaries when there is room around the focus.
    separators = "。！？!?\n"
    for position in range(start, min(relative, end)):
        if source["text"][position] in separators:
            start = position + 1
            break
    focus_end = min(len(source["text"]), meta["focus_end"] - meta["start_index"])
    for position in range(end - 1, max(start, focus_end - 1) - 1, -1):
        if source["text"][position] in separators:
            end = position + 1
            break
    return source["text"][start:end], start


def crop_metadata(source, shift):
    meta = source["metadata"]
    if "focus_start" not in meta:
        return
    meta["start_index"] += shift
    meta["end_index"] = meta["start_index"] + len(source["text"])
    if "time_mentions" in meta:
        meta["time_mentions"] = [m for m in meta["time_mentions"] if m["text"] in source["text"]]
    meta["source_chunks"] = [c for c in meta["source_chunks"]
                            if c["start_index"] < meta["end_index"] and c["end_index"] > meta["start_index"]]
    meta["source_chunk_ids"] = [c["chunk_id"] for c in meta["source_chunks"]]
    if meta["source_chunks"]:
        meta["chunk_index"] = meta["source_chunks"][0]["chunk_index"]
        meta["chunk_index_end"] = meta["source_chunks"][-1]["chunk_index"]
    meta["paragraphs"] = [p for p in meta.get("paragraphs", [])
                          if p["start_index"] < meta["end_index"] and p["end_index"] > meta["start_index"]]
    indices = [p["paragraph_index"] for p in meta["paragraphs"]]
    meta["paragraph_start"] = min(indices) if indices else None
    meta["paragraph_end"] = max(indices) if indices else None
    _refresh_table_location(meta)


def _refresh_table_location(metadata):
    if "sheet_name" not in metadata:
        return
    rows = [p for p in metadata.get("paragraphs", []) if "row_index" in p]
    metadata.update(row_start=min((p["row_index"] for p in rows), default=None),
                    row_end=max((p["row_index"] for p in rows), default=None),
                    cell_ranges=[p["cell_range"] for p in rows])
