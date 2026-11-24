"""Persistent parent ranges and extractive section summaries over original text.

Children remain the existing embedded chunks. Parents and summary sentences are
character spans, not additional vectors or generated facts. No LLM is loaded.
"""

from bisect import bisect_right
from collections import Counter
from hashlib import sha256
import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

VERSION = 2
HEADING = re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]+\S[^\n]{0,119}|第[零〇一二三四五六七八九十百千万两\d]{1,12}[章回节卷部][^\n]{0,119}|Chapter[ \t]+\d+[^\n]{0,119})[ \t]*$", re.I)
SENTENCE = re.compile(r"[^。！？!?\n]+(?:[。！？!?]+|$)", re.M)
STOP = {"什么", "哪些", "怎么", "如何", "这个", "那个", "一下", "总结", "概括", "分别", "主要", "做了", "了些", "些什", "么事", "事情", "介绍", "说明", "文档", "内容", "可以", "进行", "以及", "还是", "the", "and", "how", "what"}


def terms(text):
    result = set()
    for word in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            result.update(word[n:n + 2] for n in range(len(word) - 1))
        elif len(word) > 1:
            result.add(word)
    return result - STOP


def sentence_spans(text, start=0, end=None, max_length=180):
    """Return contiguous source spans; very long unpunctuated sentences are split."""
    end = len(text) if end is None else end
    spans = []
    for found in SENTENCE.finditer(text, start, end):
        lo, hi = found.span()
        while lo < hi and text[lo].isspace():
            lo += 1
        for position in range(lo, hi, max_length):
            stop = min(position + max_length, hi)
            while stop > position and text[stop - 1].isspace():
                stop -= 1
            if stop - position >= 4:
                spans.append({"start_index": position, "end_index": stop})
    return spans


def extract_summary(text, start, end, question="", max_sentences=3):
    """Rank source sentences by query coverage and within-section term frequency.

    Overview summaries stored on disk are query independent. At retrieval time a
    focused extract can select other original sentences from the same section.
    A sentence already mostly covered by another is not selected twice.
    """
    spans = sentence_spans(text, start, end)
    if not spans:
        return []
    tokens = [terms(text[s["start_index"]:s["end_index"]]) for s in spans]
    frequencies = Counter(token for row in tokens for token in row)
    query = terms(question)
    scored = []
    for i, (span, row) in enumerate(zip(spans, tokens)):
        relevance = len(row & query)
        centrality = sum(min(frequencies[t], 5) for t in row) / max(len(row), 1)
        # Favor informative sentences over very short high-density query matches.
        information = min((span["end_index"] - span["start_index"]) / 60, 1)
        scored.append(((10 * relevance + centrality) * information + (0.2 if i == 0 else 0), i))
    selected, used = [], []
    for _, i in sorted(scored, key=lambda item: (-item[0], item[1])):
        if any(tokens[i] and len(tokens[i] & prior) / len(tokens[i] | prior) > .85 for prior in used):
            continue
        selected.append(spans[i])
        used.append(tokens[i])
        if len(selected) >= max_sentences:
            break
    return sorted(selected, key=lambda s: s["start_index"])


def fingerprint(record, text, parent_characters):
    positions = [(c["chunk_id"], c["text"], c["metadata"]) for c in record["chunks"]]
    payload = json.dumps([VERSION, parent_characters, positions], ensure_ascii=False, sort_keys=True)
    return sha256((payload + "\n" + text).encode("utf-8")).hexdigest()


def build_hierarchy(record, text, parent_characters=2100):
    chunks = record["chunks"]
    if any(type(c["metadata"].get("start_index")) is not int
           or type(c["metadata"].get("end_index")) is not int
           or not 0 <= c["metadata"]["start_index"] < c["metadata"]["end_index"] <= len(text)
           or text[c["metadata"]["start_index"]:c["metadata"]["end_index"]] != c["text"]
           for c in chunks):
        raise ValueError("Chunk positions do not map to the saved original text")
    # Physical page/heading units are hard boundaries. Plain TXT also recognizes
    # Chinese chapter headings and English Chapter headings without changing chunks.
    boundaries = {0: ("正文", "document")}
    seen_units = set()
    for chunk in chunks:
        meta = chunk["metadata"]
        unit = meta.get("unit_index", meta.get("page_number"))
        if unit in seen_units:
            continue
        seen_units.add(unit)
        position = meta["start_index"]
        title = meta.get("section_title")
        if title or meta.get("page_number"):
            boundaries[position] = (title or f"第{meta['page_number']}页", "heading" if title else "page")
    for found in HEADING.finditer(text):
        boundaries[found.start()] = (found.group().strip().lstrip("#").strip(), "heading")
    starts = sorted(boundaries)
    sections, parents, children = [], [], {}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_characters, chunk_overlap=min(200, parent_characters // 10),
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        keep_separator="end", add_start_index=True, strip_whitespace=True,
    )
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        if not text[start:end].strip():
            continue
        section_id = f"{record['document_id']}:section:{start}"
        summary_spans = extract_summary(text, start, end)
        section = {"section_id": section_id, "title": boundaries[start][0],
                   "boundary_type": boundaries[start][1], "start_index": start, "end_index": end,
                   "summary_method": "extractive-v2", "summary_spans": summary_spans,
                   "summary": "\n".join(text[s["start_index"]:s["end_index"]] for s in summary_spans)}
        sections.append(section)
        for part in splitter.create_documents([text[start:end]]):
            lo = start + part.metadata["start_index"]
            hi = lo + len(part.page_content)
            if text[lo:hi] != part.page_content:
                raise ValueError("Parent cannot be mapped to original text")
            parent_id = f"{record['document_id']}:parent:{lo}:{hi}"
            parents.append({"parent_id": parent_id, "section_id": section_id,
                            "start_index": lo, "end_index": hi, "child_ids": []})
    # Only overlapping parents are visited (ordered spans avoid an N*M scan).
    parent_starts = [p["start_index"] for p in parents]
    for chunk in chunks:
        lo, hi = chunk["metadata"]["start_index"], chunk["metadata"]["end_index"]
        stop = bisect_right(parent_starts, hi - 1)
        hits = []
        for parent_index in range(stop - 1, -1, -1):
            p = parents[parent_index]
            if p["end_index"] <= lo:
                break
            p["child_ids"].append(chunk["chunk_id"])
            hits.append(p["parent_id"])
        children[chunk["chunk_id"]] = list(reversed(hits))
    return {"version": VERSION, "document_id": record["document_id"],
            "fingerprint": fingerprint(record, text, parent_characters),
            "parent_characters": parent_characters, "sections": sections,
            "parents": parents, "children": children}
