"""Book-wide sparse/dense fusion and source-grounded multi-chapter context planning."""
from collections import defaultdict
from copy import deepcopy
import re

from local_rag.documents.hierarchy import terms
from local_rag.documents.knowledge import TIME, NEGATION
from local_rag.rag.context import source_span


def analyze_question(question, catalog):
    people, ambiguous, variants = [], [], [question]
    for doc_id, book in catalog.items():
        for alias in book["ambiguous_aliases"]:
            if alias in question:
                ambiguous.append({"document_id": doc_id, "alias": alias})
        for entity in book["entities"]:
            names = [entity["name"], *entity["aliases"], *entity.get("retrieval_aliases", [])]
            matched = [name for name in names if name in question and name not in book["ambiguous_aliases"]]
            if matched:
                people.append({"document_id": doc_id, **entity, "matched_names": matched})
                variants.append(question + " " + " ".join(names[:8]))
    # Bounded lexical subquestions. No extra embedding or generation models.
    parts = [part.strip() for part in re.split(r"[？?；;]|以及|并且|分别|从|到|相比|对比", question) if part.strip()]
    for part in parts[:4]:
        if len(terms(part)) >= 2:
            variants.append(part)
    for person in people[:4]:
        variants.append(person["name"] + " " + " ".join(person["aliases"][:5]) + " " + question)
    return {"entities": people, "ambiguous_aliases": ambiguous,
            "queries": list(dict.fromkeys(variants))[:8]}


def reciprocal_rank_fusion(lists, limit, k=60):
    """RRF(d) = sum(1/(k+rank)); duplicate IDs vote once per input ranking."""
    totals, found, channels = defaultdict(float), {}, defaultdict(set)
    for channel, matches in lists:
        seen = set()
        for rank, match in enumerate(matches, 1):
            key = match["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            totals[key] += 1 / (k + rank)
            channels[key].add(channel)
            if key not in found or channel == "semantic":
                found[key] = deepcopy(match)
    ceiling = max(1, len(lists)) / (k + 1)
    result = []
    for key in sorted(totals, key=lambda key: (-totals[key], key))[:limit]:
        item = found[key]
        item["rrf_score"] = totals[key]
        item["score"] = totals[key] / ceiling
        item["distance"] = 1 - item["score"]
        item["metadata"].update(retrieval_channels=sorted(channels[key]), score_kind="rrf_normalized")
        result.append(item)
    return result


def diversify_chapters(matches, documents, limit):
    """Select across whole-book chapters before repeating a section."""
    groups = defaultdict(list)
    for match in matches:
        doc_id = match["metadata"]["document_id"]
        section = match["metadata"].get("section_title") or ""
        if doc_id in documents:
            sections = documents[doc_id][2]["sections"]
            position = match["metadata"].get("start_index", 0)
            section = next((s["section_id"] for s in sections if s["start_index"] <= position < s["end_index"]), section)
        groups[(doc_id, section)].append(match)
    result = []
    while groups and len(result) < limit:
        for key in list(groups):
            result.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(result) == limit:
                break
    return result


def knowledge_contexts(question, analysis, catalog, documents, plan, limit):
    """Return original contiguous spans; relationship labels remain clues, not facts."""
    if plan.scene not in {"relationship", "timeline", "cross_chapter", "alias", "overview", "comparison"}:
        return []
    grouped = defaultdict(set)
    for entity in analysis["entities"]:
        grouped[entity["document_id"]].add(entity["entity_id"])
    ranked = []
    query_terms = terms(question)
    for doc_id, wanted in grouped.items():
        if doc_id not in documents:
            continue
        record, text, hierarchy = documents[doc_id]
        book = catalog[doc_id]
        for fact in ([] if plan.scene == "alias" else book["facts"]):
            hits = wanted.intersection(fact["entity_ids"])
            if not hits:
                continue
            if plan.scene == "relationship" and (not fact["relation_clues"] or len(hits) < len(wanted)):
                continue
            if plan.scene == "timeline" and not fact["event_candidate"]:
                continue
            excerpt = text[fact["start_index"]:fact["end_index"]]
            relevance = 3 * len(hits) + len(terms(excerpt + fact["section_title"]) & query_terms)
            if plan.scene == "relationship":
                relevance += 5 * (len(hits) == len(wanted))
                relevance += 8 * ("kinship_clue" in fact["relation_clues"])
            names = [n for e in analysis["entities"] if e["document_id"] == doc_id
                     for n in [e["name"], *e["aliases"], *e.get("retrieval_aliases", [])]]
            if any(n in fact["section_title"] for n in names):
                relevance += 6
            if plan.scene == "timeline":
                relevance += 3 * any(t["kind"] == "explicit" for t in fact["time_mentions"])
            ranked.append((relevance, doc_id, fact))
        if plan.scene == "alias":
            for entity in analysis["entities"]:
                if entity["document_id"] != doc_id:
                    continue
                for declaration in entity["declarations"]:
                    section = next((s for s in hierarchy["sections"] if s["start_index"] <= declaration["start_index"] < s["end_index"]), None)
                    if section:
                        ranked.append((100 + (declaration["end_index"] - declaration["start_index"]) / 1000, doc_id, {**declaration, "section_id": section["section_id"],
                                       "section_title": section["title"], "section_order": hierarchy["sections"].index(section) + 1,
                                       "entity_ids": [entity["entity_id"]], "relation_clues": [], "time_mentions": []}))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]["start_index"]))
    # One strong original sentence per chapter, with round-robin document coverage.
    chapters = {}
    for row in ranked:
        chapters.setdefault((row[1], row[2]["section_id"]), row)
    chosen = list(chapters.values())
    if plan.scene == "timeline":
        focused = [row for row in chosen if any(
            n in row[2]["section_title"] for e in analysis["entities"] if e["document_id"] == row[1]
            for n in [e["name"], *e["aliases"], *e.get("retrieval_aliases", [])])]
        if len(focused) >= limit:
            # Prefer chapters about the person to late mentions in retrospectives.
            first = min(chosen, key=lambda row: (row[1], row[2]["section_order"]))
            chosen = focused if first in focused else [first, *focused]
        chosen.sort(key=lambda row: (row[1], row[2]["section_order"], row[2]["start_index"]))
        if len(chosen) > limit:
            dated = [i for i, row in enumerate(chosen) if any(t["kind"] == "explicit" for t in row[2]["time_mentions"])]
            indices = list(dict.fromkeys([0, len(chosen) - 1, *dated]))[:limit]
            remaining = [i for i in range(len(chosen)) if i not in indices]
            count = limit - len(indices)
            if count:
                indices += [remaining[round(i * (len(remaining) - 1) / max(1, count - 1))] for i in range(count)]
            chosen = [chosen[i] for i in sorted(indices)]
    else:
        chosen = chosen[:limit]
    result = []
    for _, doc_id, fact in chosen:
        record, text, hierarchy = documents[doc_id]
        section = next(s for s in hierarchy["sections"] if s["section_id"] == fact["section_id"])
        anchor = next((c for c in record["chunks"] if c["metadata"]["start_index"] <= fact["start_index"] < c["metadata"]["end_index"]), None)
        if anchor is None:
            continue
        anchor = {**anchor, "score": .9, "distance": .1, "model_version": None}
        start, end = fact["start_index"], fact["end_index"]
        if plan.scene != "alias":
            # Keep nearby dates, pronouns, qualifications and outcomes in the same
            # paragraph. Bound long paragraphs; never cross the source chapter.
            boundary = text.rfind("\n\n", section["start_index"], start)
            start = max(section["start_index"], boundary + 2 if boundary >= 0 else section["start_index"],
                        fact["start_index"] - 180)
            boundary = text.find("\n\n", end, section["end_index"])
            end = min(section["end_index"], boundary if boundary >= 0 else section["end_index"],
                      max(fact["end_index"], start + 600))
        source = source_span(anchor, record, text, section, start, end, "knowledge")
        if source:
            excerpt = source["text"]
            times = [{"text": m.group(), "kind": "relative" if re.match(r"[次翌当]", m.group()) else "explicit"}
                     for m in TIME.finditer(excerpt)]
            source["metadata"].update(analysis_type=plan.scene, relation_clues=fact["relation_clues"],
                                      time_mentions=times, section_order=fact["section_order"],
                                      ordering="document_narrative_order", negated=bool(NEGATION.search(excerpt)),
                                      focus_start=fact["start_index"], focus_end=fact["end_index"],
                                      score_kind="structured_candidate")
            result.append(source)
    return result


def combine_contexts(structured, expanded, limit, scene):
    # Timeline needs a sequence. Other scenes retain semantic/parent context too.
    if scene in {"timeline", "alias"} and structured:
        return structured[:limit]
    slots = max(1, limit // 2) if structured else 0
    initial = structured[:slots]
    candidates = expanded + structured[slots:]
    covered = {(s["metadata"]["document_id"], s["metadata"].get("section_id")) for s in initial}
    diverse, repeated = [], []
    for source in candidates:
        key = (source["metadata"]["document_id"], source["metadata"].get("section_id"))
        if key in covered:
            repeated.append(source)
        else:
            diverse.append(source)
            covered.add(key)
    candidates = initial + diverse + repeated
    result, seen = [], set()
    for source in candidates:
        key = (source["metadata"]["document_id"], source["metadata"].get("start_index"), source["text"])
        if key not in seen:
            result.append(source)
            seen.add(key)
        if len(result) == limit:
            break
    return result
