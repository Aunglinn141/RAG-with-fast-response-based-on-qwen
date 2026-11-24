"""Supply catalog evidence to the shared LLM workflow, never a prewritten answer."""

import json
import re

from local_rag.artifacts.models import LABELS, SOURCE_LABELS
from local_rag.artifacts.service import name_key, question_fields, resolve_facts

STATUS_LABELS = {"documented": "正式档案记载", "guide_only": "馆方讲解记载",
                 "secondary_only": "二次汇编记载", "conflict": "来源冲突，不可择一认定",
                 "missing": "未登记", "source_missing": "来源已失效"}


def catalog_contexts(service, question, *, document_id=None, limit=3, exact_only=False):
    """Respect document scope, source liveness and ambiguous identities.

    Structured entries are atomic context units: the RAG budgeter must keep or
    omit each entire entry so conflict and identity qualifications cannot be lost.
    """
    with service.lock:
        records = []
        for row in service.registry.rows():
            if row["index_status"] == "deleting":
                continue
            record = service._project(row)
            # Filter assertions before resolving fields: a scoped request must not
            # import museum/era claims from a different document for the same item.
            claims = [c for f in record["facts"].values() for c in f["claims"]
                      if c["available"] and (document_id is None or c["document_id"] == document_id)]
            if not any(c["field"] == "name" for c in claims):
                continue
            facts = resolve_facts(claims)
            records.append({**record, "name": next(c["value"] for c in claims if c["field"] == "name"),
                            "facts": facts, "claims": claims,
                            "aliases": [c["value"] for c in claims if c["field"] == "alias"]})
        if not records:
            return []
        query = name_key(question)
        hits = []
        for r in records:
            names = [r["artifact_id"], *r["aliases"], *[c["value"] for c in r["claims"] if c["field"] == "name"]]
            # A quoted work title remains a valid mention without the creator
            # prefix or scroll suffix. Shared titles still retain ambiguity.
            names += [title for name in names for title in re.findall(r'《([^》]{2,60})》', name)]
            found = [name_key(n) for n in names if name_key(n) in query]
            if found:
                hits.append((r, max(found, key=len)))
        exact = [(r, n) for r, n in hits if not any(n != other and n in other for _, other in hits)]
        named_museums = {c["value"] for r, _ in exact for c in r["claims"]
                         if c["field"] == "museum" and name_key(c["value"]) in query}
        if named_museums:
            exact = [(r, n) for r, n in exact if any(c["field"] == "museum" and c["value"] in named_museums
                                                    for c in r["claims"])]
        ambiguous = len({r["artifact_id"] for r, _ in exact}) > len({n for _, n in exact})
        resolution = "同名或别名对应多件文物，需要询问收藏馆或编号" if ambiguous else "名称或系统编号匹配"
        selected = [r for r, _ in exact]
        candidate_only = not selected
        if not selected:
            if exact_only:
                return []
            # Questions about the uploaded publications belong to document QA,
            # even if their titles mention museums. Catalog candidates would
            # crowd out the reading notes that establish source provenance.
            if re.search(r"资料|图录|文件|文档|汇编|\bdocuments?\b|\bpublications?\b", question, re.I):
                return []
            # General document QA should not be diverted into unrelated catalog entries.
            if document_id is not None or not re.search(r"文物|馆藏|博物馆|收藏|器物|青铜器|^(?:它|这件|该文物)|\bartifact\b|\bmuseum\b", question, re.I):
                return []
            candidates = service.search(question, limit=limit)["candidates"]
            ids = {c["artifact_id"] for c in candidates}
            selected = [r for r in records if r["artifact_id"] in ids]
            resolution = "仅为检索候选，身份未确定，需要用户明确名称或编号"
        if not selected:
            return []
        requested_fields = question_fields(question)
        fields = list(dict.fromkeys(["name", *(["museum"] if ambiguous or candidate_only else []), *requested_fields]))
        if candidate_only:
            # Until the user identifies the object, provide only the candidate
            # names and museums for clarification, not facts about a guessed item.
            fields = ["name", "museum"]
        contexts = []
        for record in selected[:limit]:
            facts, evidence = {}, []
            description = record["facts"]["description"]
            has_description = not candidate_only and bool(description["claims"]) and all(
                len(c["value"]) <= 500 for c in description["claims"])
            # A field lookup needs only those fields. Irrelevant narrative can
            # cause a small model to mix collection, excavation and exhibition.
            narrative_requested = bool(re.search(
                r'介绍|解释|描述|比较|对比|区别|外形|形状|形制|尺寸|长宽|口径|高度|重量|容积|铭文|款识|结构|组合|工艺|制作|出土|展厅|展出|今天|目前|现在|序号|NO\.|\b(?:describe|compare|shape|size|inscription|construction|today|currently)\b',
                question, re.I))
            field_lookup = bool(requested_fields) and not candidate_only and not narrative_requested and all(
                record['facts'][f]['claims'] for f in requested_fields)
            has_description = has_description and not field_lookup
            # Narrative details (dimensions, inscriptions, construction) are tied
            # to this object. A generic page summary can select its neighbor instead.
            record_fields = [*fields, "description"] if has_description else fields
            for field in record_fields:
                fact = record["facts"][field]
                active = fact["claims"]
                if not active and has_description and field != 'accession_number':
                    # An absent structured field is not a statement that the
                    # original narrative lacks that fact. Avoid contradictory
                    # "unknown purpose" answers beside an explicit description.
                    continue
                # A long fact is withheld rather than silently truncated into a
                # different claim. Its state (especially conflict) remains visible.
                values = [{"value": c["value"] if len(c["value"]) <= 500 else "原文较长，此处省略，不能据此补全",
                           **({"source_type": SOURCE_LABELS[c["source_type"]]}
                              if fact['status'] == 'conflict' or len({v['source_type'] for v in active}) > 1 else {})}
                          for c in active]
                facts[LABELS[field]] = {"status": STATUS_LABELS[fact["status"]], "statements": values}
                evidence.extend(active)
            # Keep the internal UUID in metadata, out of generation evidence: it
            # is not a museum accession number and must never become one.
            # Preserve a short original heading when it directly precedes the
            # name, so editorial labels remain visible without becoming facts.
            headings = []
            for claim in record["claims"] if re.search(r'编号|序号|NO\.|\baccession\b', question, re.I) else []:
                if claim["field"] != "name":
                    continue
                lines = claim["quote"].splitlines()
                for position, line in enumerate(lines[:4]):
                    if line.strip() == claim["value"] and position and sum(map(len, lines[:position + 1])) <= 160:
                        headings.append("\n".join(lines[:position + 1]))
                        break
            text = json.dumps({"身份判定": resolution,
                               **({"来源说明": "二次汇编，未经馆方原档案核验"}
                                  if any(c["source_type"] == "secondary" for c in evidence) else {}),
                               **({"原文条目标题": list(dict.fromkeys(headings))} if headings and not candidate_only else {}),
                               **({"标题用途": "仅辅助识别原文条目，不能据此认定正式馆藏编号"}
                                  if headings and not candidate_only else {}),
                               "匹配记录总数": len(selected), "字段": facts}, ensure_ascii=False, separators=(",", ":"))
            anchor = record["claims"][0]
            contexts.append({"chunk_id": f"artifact:{record['artifact_id']}:{record['revision']}", "text": text,
                "score": 0.0, "distance": 1.0, "model_version": "structured-artifacts-v1",
                "metadata": {"context_type": "artifact", "artifact_id": record["artifact_id"],
                             "revision": record["revision"], "document_id": anchor["document_id"],
                             "filename": anchor["filename"], "section_title": "文物结构化证据",
                             "derived": True, "atomic": True, "score_kind": "catalog_evidence",
                             "resolution": resolution, "matched_records": len(selected),
                             "has_description": has_description,
                             "complete_catalog": has_description or field_lookup,
                             "claims": evidence}})
        return contexts


def conversation_catalog_query(service, question, history, *, document_id=None):
    """Resolve known artifact references from this chat's most recent mentions.

    Only the identity is carried forward. Facts are retrieved again from live
    sources. A newly named object wins over old mentions; same-name ambiguity
    must survive resolution instead of turning into a fabricated unique ID.
    """
    if not history:
        return None
    if catalog_contexts(service, question, document_id=document_id, exact_only=True):
        return question, "catalog_explicit", len(history)
    if not re.match(r"^(?:那么|那|所以|请问|请|再|能否|可以|还|也|\s)*(?:它|这件|该文物|上述|刚才|是?什么|在哪|保存在哪|收藏在哪|有何|有什么|有啥|用途|材料|年代|文化价值|详细|介绍一下|继续)|"
                    r"^(?:what|where|when|why|how|is|does|can|tell|and)\b.*\b(?:it|its|that|this)\b", question, re.I):
        return None
    # Comparisons and references to older objects require the full LLM history.
    if re.search(r"第[一二三四五六七八九十\d]|前一|上一个|上一件|比较|对比|两[件个]|\b(?:first|previous|both|compare)\b", question, re.I):
        return None
    for turn in reversed(history):
        for text in (turn["question"], turn["answer"]):
            contexts = catalog_contexts(service, text, document_id=document_id, exact_only=True)
            if not contexts:
                continue
            if len(contexts) == 1 and contexts[0]["metadata"]["matched_records"] == 1:
                identities = [contexts[0]["metadata"]["artifact_id"]]
            else:
                identities = list(dict.fromkeys(claim["value"] for source in contexts
                    for claim in source["metadata"]["claims"] if claim["field"] == "name"))
            hint = "、".join(identities)
            resolved = f"{hint}：{question}"
            return (resolved, "catalog_history", len(history)) if len(resolved) <= 2000 else None
    return None
