"""Select useful context from a global candidate pool without loading another model."""

from collections import Counter


def select_context_candidates(matches, limit):
    """Deduplicate exact text, then balance relevance with document coverage.

    A small, capped same-document penalty lets similarly relevant documents
    contribute without forcing a much less relevant document into the answer.
    Input scores, text and source metadata are never rewritten.
    """
    candidates, seen_ids, seen_text = [], set(), set()
    for position, match in enumerate(matches):
        text_key = " ".join(match.get("text", "").split())
        chunk_id = match["chunk_id"]
        if not text_key or chunk_id in seen_ids or text_key in seen_text:
            continue
        seen_ids.add(chunk_id)
        seen_text.add(text_key)
        document_id = match.get("metadata", {}).get("document_id") or chunk_id
        candidates.append((position, match, document_id))
    chosen, counts = [], Counter()
    while candidates and len(chosen) < limit:
        best = max(range(len(candidates)), key=lambda i: (
            candidates[i][1]["score"] - 0.04 * min(counts[candidates[i][2]], 3),
            -candidates[i][0],
        ))
        _, match, document_id = candidates.pop(best)
        chosen.append(match)
        counts[document_id] += 1
    return chosen
