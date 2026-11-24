"""Coordinate evidence, relational facts, isolated BGE vectors, and safe answers."""

import hashlib
import json
import logging
import re
from uuid import uuid4

from local_rag.artifacts.models import CORE_FIELDS, LABELS, SOURCE_RANK
from local_rag.core.exceptions import DocumentError
from local_rag.embeddings.bge import validate_vectors
from local_rag.storage.artifacts import ArtifactRegistry
from local_rag.storage.chroma import ChromaStore

logger = logging.getLogger(__name__)


def compact(text):
    return re.sub(r"\s+", "", text).casefold()


def name_key(text):
    return re.sub(r"[\s《》〈〉“”\"'‘’]", "", text).casefold()


def resolve_facts(claims):
    facts = {}
    for field in LABELS:
        evidence = sorted((c for c in claims if c["field"] == field), key=lambda c: SOURCE_RANK[c["source_type"]])
        active = [c for c in evidence if c.get("available", True)]
        values = {compact(c["value"]) for c in active}
        if not active:
            status, value = ("source_missing" if evidence else "missing"), None
        elif len(values) > 1 and field != "alias":
            status, value = "conflict", None
        else:
            status = {"official_archive": "documented", "museum_guide": "guide_only", "secondary": "secondary_only"}[active[0]["source_type"]]
            value = active[0]["value"]
        facts[field] = {"value": value, "status": status, "claims": evidence}
    return facts


class ArtifactVectors(ChromaStore):
    PREFIX = "localrag_artifacts_"


class ArtifactService:
    def __init__(self, index):
        self.index = index
        self.lock = index.lock  # Sources cannot disappear during a catalog transaction.
        with self.lock:
            self.registry = ArtifactRegistry(index.settings.registry_path.with_suffix(".artifacts.sqlite3"))
        self.vectors = ArtifactVectors(index.settings.chroma_dir / "artifacts")

    def _validated_claims(self, inputs):
        documents, result, seen = {}, [], set()
        for item in inputs:
            document_id = str(item.document_id)
            if document_id not in documents:
                record = self.index.get(document_id)
                if record["status"] in {"deleting", "delete_failed", "deleted"}:
                    raise DocumentError("artifact_source_unavailable", "来源文档已退出知识库。", 409)
                text = (self.index.documents.directory(document_id) / "text.txt").read_text(encoding="utf-8")
                positions = [i for i, c in enumerate(text) if not c.isspace()]
                normalized = "".join(text[i] for i in positions)
                documents[document_id] = record, text, positions, normalized
            record, text, positions, normalized = documents[document_id]
            if item.source_type != "secondary" and re.search(r"非馆方原版|不是馆方出版|编者导读（非馆方", text):
                raise DocumentError("artifact_source_type_mismatch", "来源明确标注为编者汇编，请使用“二次汇编”类型，不能登记为馆方原档案。", 422)
            needle = re.sub(r"\s+", "", item.quote)
            value = re.sub(r"\s+", "", item.value)
            if not needle or value not in needle:
                raise DocumentError("artifact_value_not_in_quote", f"{LABELS[item.field]}的值必须逐字出现在来源摘录中（仅忽略空白）。", 422)
            if item.start_index is not None:
                start = item.start_index
                offset = next((i for i, p in enumerate(positions) if p >= start), len(positions))
                found = offset if normalized.startswith(needle, offset) and offset < len(positions) and positions[offset] == start else -1
            else:
                found = normalized.find(needle)
                if found >= 0 and normalized.find(needle, found + 1) >= 0:
                    raise DocumentError("artifact_quote_ambiguous", "摘录在文档中出现多次，请提供更完整的原文或准确的 start_index。", 422)
            if found < 0:
                raise DocumentError("artifact_quote_not_found", "来源摘录与保存的原文不一致，档案未保存。", 422)
            start, end = positions[found], positions[found + len(needle) - 1] + 1
            chunks = [c for c in record["chunks"] if c["metadata"].get("start_index", 0) < end
                      and c["metadata"].get("end_index", 0) > start]
            claim = {"field": item.field, "value": item.value, "document_id": document_id,
                     "quote": text[start:end], "source_type": item.source_type,
                     "filename": record["filename"], "file_hash": record["sha256"],
                     "start_index": start, "end_index": end,
                     "page_numbers": sorted({c["metadata"]["page_number"] for c in chunks if c["metadata"].get("page_number")}),
                     "chunk_ids": [c["chunk_id"] for c in chunks]}
            key = json.dumps(claim, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                result.append(claim)
                seen.add(key)
        return result

    def _project(self, row):
        claims = self.registry.claims(row["artifact_id"])
        for claim in claims:
            source = self.index.registry.get(claim["document_id"])
            claim["available"] = bool(source and source["status"] not in {"deleted", "deleting", "delete_failed"}
                                      and source["file_hash"] == claim["file_hash"]
                                      and (self.index.documents.root / claim["document_id"] / "text.txt").is_file())
        return {"artifact_id": row["artifact_id"], "revision": row["revision"], "name": row["name"],
                "facts": resolve_facts(claims), "aliases": list(dict.fromkeys(c["value"] for c in claims if c["field"] == "alias" and c["available"])),
                "index_status": row["index_status"], "index_error": row["index_error"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def get(self, artifact_id):
        with self.lock:
            return self._project(self.registry.get(artifact_id))

    def list(self, query="", museum=None, offset=0, limit=100):
        with self.lock:
            all_records = [self._project(r) for r in self.registry.rows()]
            museums = sorted({r["facts"]["museum"]["value"] for r in all_records if r["facts"]["museum"]["value"]})
            records = [r for r in all_records if (not museum or r["facts"]["museum"]["value"] == museum)
                       and (not query or name_key(query) in name_key(" ".join([r["artifact_id"], r["name"], *r["aliases"],
                            *[f["value"] or "" for f in r["facts"].values()]])))]
            return {"artifacts": records[offset:offset + limit], "count": len(records), "total": len(all_records), "museums": museums}

    def save(self, request, artifact_id=None):
        with self.lock:
            claims = self._validated_claims(request.claims)
            facts = resolve_facts(claims)
            fields = {key: value["value"] for key, value in facts.items()}
            fields["name"] = fields["name"] or next(c["value"] for c in claims if c["field"] == "name")
            payload = sorted(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in claims)
            digest = hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()
            artifact_id, changed = self.registry.save(claims, fields, digest, artifact_id, request.expected_revision)
            if request.build_index and (changed or self.registry.get(artifact_id)["index_status"] != "indexed"):
                return self.reindex(artifact_id)
            return self.get(artifact_id)

    def import_records(self, request):
        # Validate every source before the first write. Report remaining per-record errors explicitly.
        with self.lock:
            for item in request.records:
                self._validated_claims(item.claims)
            results = []
            for number, item in enumerate(request.records, 1):
                try:
                    results.append({"record": number, "artifact": self.save(item), "error": None})
                except DocumentError as exc:
                    results.append({"record": number, "artifact": None, "error": {"code": exc.code, "message": str(exc)}})
            return {"results": results, "saved": sum(r["artifact"] is not None for r in results),
                    "failed": sum(r["error"] is not None for r in results)}

    def reindex(self, artifact_id):
        with self.lock:
            record = self.get(artifact_id)
            self.registry.index_state(artifact_id, "indexing")
            try:
                self.vectors.delete_document(artifact_id)
                facts = record["facts"]
                if any(not c["available"] for fact in facts.values() for c in fact["claims"]):
                    raise ValueError("来源已删除或发生变化，请更新档案来源后重建索引。")
                identity = self.index.embedder.identity()
                if identity.dimension != 1024:
                    raise ValueError("BGE-M3 向量维度必须为 1024。")
                collection = self.vectors.prepare(identity)
                generation = uuid4().hex
                # One entry per field, always carrying the artifact ID and its exact provenance.
                chunks = []
                prefix = f"文物编号：{artifact_id}\n名称：{record['name']}\n收藏博物馆：{facts['museum']['value'] or '暂无资料'}"
                for field, fact in facts.items():
                    if not fact["value"]:
                        continue
                    chunks.append({"chunk_id": f"{artifact_id}:{len(chunks) + 1}",
                                   "text": f"{prefix}\n{LABELS[field]}：{fact['value']}",
                                   "metadata": {"artifact_id": artifact_id, "field": field, "revision": record["revision"],
                                                "chunk_index": len(chunks) + 1, "claims": fact["claims"]}})
                if not chunks:
                    raise ValueError("没有无冲突的字段可供索引。")
                for start in range(0, len(chunks), self.index.settings.batch_size):
                    batch = chunks[start:start + self.index.settings.batch_size]
                    vectors = self.index.embedder.embed_documents([c["text"] for c in batch])
                    validate_vectors(vectors, len(batch), 1024)
                    if self.index.embedder.identity() != identity:
                        raise ValueError("索引期间模型版本变化，请重建。")
                    self.vectors.upsert(collection, artifact_id, generation, batch, vectors)
                if self.vectors.count_document(artifact_id) != len(chunks):
                    raise ValueError("向量数量校验失败，请重建。")
                self.registry.index_state(artifact_id, "indexed", collection=collection, generation=generation, revision=record["revision"])
            except Exception as exc:
                logger.warning("Artifact index failed for %s: %s", artifact_id, exc)
                self.registry.index_state(artifact_id, "failed", error="档案已保存，向量索引失败；请检查来源、本地 BGE-M3 和存储后重试。")
                try:
                    self.vectors.delete_document(artifact_id)
                except Exception:
                    logger.exception("Artifact vector cleanup requires retry: %s", artifact_id)
            return self.get(artifact_id)

    def invalidate_document(self, document_id):
        for artifact_id in self.registry.referencing(document_id):
            try:
                self.registry.get(artifact_id)
            except DocumentError:
                continue
            self.registry.index_state(artifact_id, "stale", error="来源文档已删除，请更新档案来源。")
            self.vectors.delete_document(artifact_id)

    def delete(self, artifact_id):
        with self.lock:
            self.registry.get(artifact_id)
            # Visibility is revoked before cleanup; a failed cleanup remains retryable.
            self.registry.index_state(artifact_id, "deleting")
            try:
                self.vectors.delete_document(artifact_id)
            except Exception as exc:
                raise DocumentError("artifact_delete_failed", "文物已退出检索，向量清理失败，请重试删除。", 503) from exc
            self.registry.index_state(artifact_id, "deleted")
            return {"artifact_id": artifact_id, "status": "deleted", "remaining_vectors": 0}

    @staticmethod
    def _candidate(record, score=None):
        return {"artifact_id": record["artifact_id"], "name": record["name"],
                "museum": record["facts"]["museum"]["value"], "era": record["facts"]["era"]["value"], "score": score}

    def search(self, query, museum=None, limit=10, semantic=True):
        with self.lock:
            rows = [r for r in self.registry.rows() if r["index_status"] != "deleting"]
            records = {r["artifact_id"]: self._project(r) for r in rows}
            records = {key: r for key, r in records.items() if not museum or r["facts"]["museum"]["value"] == museum}
            lexical = [r for r in records.values() if name_key(query) in name_key(" ".join(
                [r["artifact_id"], r["name"], *r["aliases"], *[f["value"] or "" for f in r["facts"].values()]]))]
            candidates = [self._candidate(r) for r in lexical]
            warnings = []
            compatible = [r for r in rows if r["artifact_id"] in records and r["index_status"] == "indexed"
                          and r["revision"] == r["indexed_revision"]
                          and all(c["available"] for f in records[r["artifact_id"]]["facts"].values() for c in f["claims"])]
            if semantic and compatible:
                try:
                    identity = self.index.embedder.identity()
                    collection = self.vectors.collection_name(identity)
                    current = [r for r in compatible if r["collection_name"] == collection]
                    if len(current) != len(compatible):
                        warnings.append("部分文物索引的模型版本已变更，请重建索引。")
                    if current:
                        vector = self.index.embedder.embed_query(query)
                        validate_vectors([vector], 1, identity.dimension)
                        if self.index.embedder.identity() != identity:
                            raise ValueError("Embedding model changed during search")
                        matches = self.vectors.search(collection, vector,
                            [{**r, "document_id": r["artifact_id"]} for r in current], min(500, limit * 9))
                        seen = {r["artifact_id"] for r in candidates}
                        for match in matches:
                            key = match["metadata"]["artifact_id"]
                            if key not in seen:
                                candidates.append(self._candidate(records[key], match["score"]))
                                seen.add(key)
                except Exception:
                    logger.exception("Artifact semantic search unavailable")
                    warnings.append("向量检索暂不可用，当前仅显示名称和字段匹配结果。")
            elif semantic and records and not compatible:
                warnings.append("文物尚无可用向量索引，当前仅使用名称和字段匹配。")
            return {"query": query, "candidates": candidates[:limit], "warnings": warnings}

    def ask(self, question, artifact_id=None, museum=None):
        with self.lock:
            base = {"question": question, "answer": "", "status": "insufficient_evidence", "warnings": [], "sources": []}
            if artifact_id:
                record = self.get(artifact_id)
                if record["index_status"] == "deleting" or (museum and record["facts"]["museum"]["value"] != museum):
                    return {**base, "answer": "指定文物不在当前可用的博物馆范围内。"}
            else:
                records = [self._project(r) for r in self.registry.rows() if r["index_status"] != "deleting"]
                records = [r for r in records if not museum or r["facts"]["museum"]["value"] == museum]
                question_key = name_key(question)
                matched = []
                for r in records:
                    names = [r["artifact_id"], *r["aliases"], *[c["value"] for c in r["facts"]["name"]["claims"] if c["available"]]]
                    hits = [name_key(n) for n in names if name_key(n) in question_key]
                    if hits:
                        matched.append((r, max(hits, key=len)))
                # A contained short name is not a second entity; equal names remain ambiguous.
                matches = [r for r, hit in matched if not any(hit != other and hit in other for _, other in matched)]
                if len(matches) != 1:
                    result = self.search(question, museum, 5) if not matches else {"candidates": [self._candidate(r) for r in matches], "warnings": []}
                    return {**base, "status": "needs_clarification" if result["candidates"] else "insufficient_evidence",
                            "answer": "请指定文物编号或从候选中选择一件文物；相似度不能确定文物身份。" if result["candidates"] else "未找到明确对应的文物，请提供文物名称或编号。",
                            "candidates": result["candidates"], "warnings": result["warnings"]}
                record = matches[0]
            fields = question_fields(question)
            unsupported = re.findall(r"重量|尺寸|价格|估价|多少钱|真伪|出土地点|出土地|展厅|展出地点|开放时间|修复记录", question)
            if not fields:
                return {**base, "artifact_id": record["artifact_id"],
                        "answer": "暂无可直接回答此问题的结构化资料。当前支持名称、年代、材料、用途、文化价值、收藏博物馆及馆藏编号。"}
            lines, sources, selected, answered = [], [], {}, 0
            for field in fields:
                fact = record["facts"][field]
                selected[field] = fact
                if fact["status"] == "conflict":
                    variants = []
                    for c in fact["claims"]:
                        if c["available"]:
                            sources.append(c)
                            variants.append(f"{c['value']} [{len(sources)}]")
                    lines.append(f"{LABELS[field]}：资料存在冲突，暂不能确定。" + "；".join(variants))
                elif fact["value"] is None:
                    lines.append(f"{LABELS[field]}：暂无资料。" + ("原来源已不可用，请补充档案。" if fact["status"] == "source_missing" else ""))
                else:
                    answered += 1
                    c = next(c for c in fact["claims"] if c["available"])
                    sources.append(c)
                    qualifier = {"documented": "", "guide_only": "据馆方讲解资料，", "secondary_only": "据二次汇编（尚未经馆方原档案核验），"}[fact["status"]]
                    lines.append(f"{LABELS[field]}：{qualifier}{fact['value']} [{len(sources)}]")
            if unsupported:
                lines.append("、".join(dict.fromkeys(unsupported)) + "：暂无可核对的结构化资料。")
            return {**base, "status": "answered" if answered == len(fields) and not unsupported else "partial" if answered else "insufficient_evidence",
                    "answer": "\n".join(lines), "artifact_id": record["artifact_id"], "facts": selected, "sources": sources}

    def close(self):
        self.vectors.close()


def question_fields(question):
    keys = {
        "name": r"名称|名字|叫什么|叫啥|\bname\b",
        "era": r"年代|时代|朝代|时期|什么文化|哪种文化|哪个文化|哪个朝|什么朝|什么时候|何时|\bera\b|\bdynasty\b|\bperiod\b|\bwhich culture\b",
        "material": r"材料|材质|质地|什么做|什么制|制作材料|\bmaterial",
        "purpose": r"用途|用来|用做|用作|作用|功能|干什么|做什么用|\buse\b|\bpurpose\b|\bfunction\b",
        "cultural_value": r"文化价值|历史价值|艺术价值|研究价值|意义|重要性|为什么重要|\bsignificance\b|\bvalue\b",
        "museum": r"博物|馆藏于|收藏于|收藏在哪|藏在哪|保存在哪|存放|收藏地|收藏单位|保管单位|在哪个馆|在哪里|在哪儿|哪家馆|哪个馆|\bmuseum\b|\bwhere\b",
        "accession_number": r"馆藏编号|登记号|藏品号|\baccession\b",
    }
    fields = [f for f, pattern in keys.items() if re.search(pattern, question, re.I)]
    if not fields and re.search(r"介绍|概况|基本信息|有哪些信息|了解|\boverview\b|\bintroduce\b", question, re.I):
        fields = list(CORE_FIELDS)
    return fields
