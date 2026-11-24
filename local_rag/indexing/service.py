"""Recoverable, synchronous first-version indexing for a single local API worker."""

import logging
from collections import OrderedDict
from uuid import UUID, uuid4

from filelock import FileLock

from local_rag.clients.ollama import OllamaError
from local_rag.core.exceptions import DocumentError
from local_rag.embeddings.bge import BGEEmbeddings, validate_vectors
from local_rag.storage.chroma import ChromaStore
from local_rag.storage.knowledge import KnowledgeStore
from local_rag.storage.registry import DocumentRegistry, now
from local_rag.core.timing import stage, timed_operation

logger = logging.getLogger(__name__)


class IndexService:
    def __init__(self, documents, settings, embedder=None):
        self.documents, self.settings = documents, settings
        settings.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(settings.registry_path) + ".lock", timeout=settings.lock_timeout)
        self.vectors = ChromaStore(settings.chroma_dir)
        self.embedder = embedder if embedder is not None else BGEEmbeddings(settings)
        self._book_cache = OrderedDict()
        with self.lock:
            self.registry = DocumentRegistry(settings.registry_path)
            self.knowledge = KnowledgeStore(settings.registry_path.with_suffix('.knowledge.sqlite3'))
            self._recover()

    @staticmethod
    def canonical_id(document_id):
        try:
            return str(UUID(document_id))
        except (ValueError, AttributeError) as exc:
            raise DocumentError("document_not_found", "文档不存在。", 404) from exc

    def _row(self, document_id, *, include_deleted=False):
        row = self.registry.get(self.canonical_id(document_id))
        if row is None or (row["status"] == "deleted" and not include_deleted):
            raise DocumentError("document_not_found", "文档不存在。", 404)
        return row

    def _recover(self):
        # Import step-2 documents without invoking a model or overwriting existing states.
        for path in self.documents.root.glob("*/document.json"):
            try:
                document_id = str(UUID(path.parent.name))
            except ValueError:
                continue
            self.registry.register(self.documents.get(document_id))
        for row in self.registry.rows():
            document_id = row["document_id"]
            try:
                if row["status"] in {"indexing", "cleanup_required"}:
                    self.knowledge.delete(document_id)
                    self.vectors.delete_document(document_id)
                    self.registry.update(document_id, status="failed", generation=None,
                                         last_error="上次索引中断；已清理残留，请重新索引。")
                elif row["status"] in {"deleting", "delete_failed"}:
                    self._delete(document_id)
            except Exception:
                # Incomplete documents remain invisible. A later restart/retry can finish cleanup.
                logger.exception("Recovery needs retry for document %s", document_id)

    def get(self, document_id):
        row = self._row(document_id)
        record = self.documents.get(row["document_id"])
        return {**record, "status": row["status"], "index": row}

    def list_documents(self):
        with self.lock:
            return self.registry.rows()

    def ingest(self, stream, filename, content_type):
        with self.lock:
            record = self.documents.ingest(stream, filename, content_type)
            self.registry.register(record)
            if self.settings.auto_index:
                self._index(record["document_id"])
            return self.get(record["document_id"])

    def reindex(self, document_id):
        with self.lock:
            row = self._row(document_id)
            if row["status"] in {"deleting", "delete_failed"}:
                raise DocumentError("document_deleting", "文档正在删除，请重试删除操作。", 409)
            self._index(row["document_id"])
            return self.get(row["document_id"])

    def _index(self, document_id):
        self._book_cache.pop(document_id, None)
        self.registry.update(document_id, status="indexing", generation=None, indexed_at=None, last_error=None)
        try:
            self.knowledge.delete(document_id)
            # Remove all versions first. Failed rebuilds are explicitly unavailable until retry.
            self.vectors.delete_document(document_id)
            record = self.documents.get(document_id)
            identity = self.embedder.identity()
            if identity.dimension != 1024:
                raise OllamaError("BGE-M3 向量维度必须为 1024。")
            name = self.vectors.prepare(identity)
            generation = uuid4().hex
            self.registry.update(document_id, model_name=identity.name, model_version=identity.version,
                                 embedding_dimension=identity.dimension, collection_name=name)
            chunks = record["chunks"]
            # Derived local text only; shares the document's deletion lifecycle.
            self.documents.hierarchy(document_id)
            for start in range(0, len(chunks), self.settings.batch_size):
                batch = chunks[start:start + self.settings.batch_size]
                vectors = self.embedder.embed_documents([chunk["text"] for chunk in batch])
                validate_vectors(vectors, len(batch), identity.dimension)
                if self.embedder.identity() != identity:
                    raise OllamaError("索引期间模型版本发生变化，请重新索引。")
                self.vectors.upsert(name, document_id, generation, batch, vectors)
            if self.vectors.count_document(document_id) != len(chunks):
                raise RuntimeError("Stored vector count does not match document chunk count")
            book_record, book_text, hierarchy = self.documents.hierarchy(document_id)
            self.knowledge.sync({**self.registry.get(document_id), "generation": generation}, book_record, book_text, hierarchy)
            self.registry.update(document_id, status="indexed", generation=generation,
                                 chunk_count=len(chunks), indexed_at=now(), last_error=None)
        except Exception as exc:
            status = "failed"
            try:
                self.knowledge.delete(document_id)
                self.vectors.delete_document(document_id)
            except Exception:
                status = "cleanup_required"
                logger.exception("Index cleanup needs retry for %s", document_id)
            self.registry.update(document_id, status=status, generation=None, last_error=str(exc))
            logger.warning("Index failed for %s: %s", document_id, exc)
            message = ("原文件和文本块已保留，索引失败。请检查本地 Ollama 的 bge-m3 模型，然后调用重新索引接口。"
                       if isinstance(exc, OllamaError) else "原文件和文本块已保留，索引失败。请检查本地存储和服务日志后重新索引。")
            raise DocumentError("index_failed", message, 503, document_id=document_id) from exc

    def delete(self, document_id):
        with self.lock:
            row = self._row(document_id, include_deleted=True)
            self._delete(row["document_id"])
            return {"document_id": row["document_id"], "status": "deleted", "remaining_vectors": 0}

    def _delete(self, document_id):
        self._book_cache.pop(document_id, None)
        # Commit invisibility before touching Chroma or the filesystem.
        self.registry.update(document_id, status="deleting", generation=None)
        try:
            if getattr(self, "artifacts", None) is not None:
                self.artifacts.invalidate_document(document_id)
            self.vectors.delete_document(document_id)
            self.knowledge.delete(document_id)
            self.documents.delete(document_id)
            # Keep a tombstone so retries and startup migration cannot resurrect a deleted document.
            self.registry.update(document_id, status="deleted", last_error=None)
        except Exception as exc:
            self.registry.update(document_id, status="delete_failed", last_error=str(exc))
            logger.exception("Delete needs retry for %s", document_id)
            raise DocumentError("delete_failed", "文档已退出搜索，但清理未完成；请重试删除。", 503,
                                document_id=document_id) from exc

    @timed_operation("retrieval")
    def search(self, query, limit=5, document_id=None):
        with self.lock:
            rows = [r for r in self.registry.rows() if r["status"] == "indexed"]
            if document_id is not None:
                document_id = self.canonical_id(document_id)
                rows = [r for r in rows if r["document_id"] == document_id]
            if not rows:
                return {"query": query, "matches": [], "model": None, "excluded_documents": 0}
            try:
                identity = self.embedder.identity()
                name = self.vectors.collection_name(identity)
                compatible = [r for r in rows if r["collection_name"] == name]
                if not compatible:
                    raise DocumentError("model_version_mismatch", "模型版本已变更，请先重新索引文档。", 409)
                with stage("embedding"):
                    vector = self.embedder.embed_query(query)
                vectors = [vector]
                validate_vectors(vectors, 1, identity.dimension)
                if self.embedder.identity() != identity:
                    raise DocumentError("model_version_mismatch", "搜索期间模型版本发生变化，请重试。", 409)
                with stage("chroma"):
                    matches = self.vectors.search(name, vector, compatible, limit)
            except OllamaError as exc:
                raise DocumentError("embedding_unavailable", "无法使用本地 BGE-M3，请检查 Ollama 服务和模型。", 503) from exc
            except DocumentError:
                raise
            except Exception as exc:
                logger.exception("Persistent vector search failed")
                raise DocumentError("search_failed", "向量存储读取失败，请检查服务日志及 Chroma 存储目录。", 503) from exc
            for match in matches:
                match["model_version"] = identity.version
            return {"query": query, "matches": matches,
                    "model": {"name": identity.name, "version": identity.version, "dimension": identity.dimension},
                    "excluded_documents": len(rows) - len(compatible)}

    def _book_bundle(self, row, parent_characters):
        document_id = row["document_id"]
        directory = self.documents.directory(document_id)
        def key():
            stats = []
            for filename in ("document.json", "text.txt", "retrieval.json"):
                try:
                    stat = (directory / filename).stat()
                    stats.append((stat.st_mtime_ns, stat.st_size))
                except FileNotFoundError:
                    stats.append(None)
            return (row["generation"], parent_characters, tuple(stats))
        cached = self._book_cache.get(document_id)
        if cached is not None and cached[0] == key():
            self._book_cache.move_to_end(document_id)
            return cached[1]
        bundle = self.documents.hierarchy(document_id, parent_characters)
        self.knowledge.sync(row, *bundle)
        self._book_cache[document_id] = (key(), bundle)
        self._book_cache.move_to_end(document_id)
        while len(self._book_cache) > 4:
            self._book_cache.popitem(last=False)
        return bundle

    def _book_search(self, query, limit, document_id, settings):
        from local_rag.rag.book import analyze_question, reciprocal_rank_fusion, diversify_chapters
        result = self.search(query, max(limit, 24), document_id)
        model = result["model"]
        rows = [r for r in self.registry.rows() if r["status"] == "indexed" and model
                and r["model_version"] == model["version"] and r["model_name"] == model["name"]
                and r["embedding_dimension"] == model["dimension"]
                and (document_id is None or r["document_id"] == self.canonical_id(document_id))]
        documents, valid_rows, fallback = {}, [], []
        for row in rows:
            try:
                bundle = self._book_bundle(row, settings.parent_characters)
            except ValueError:
                fallback.append(row["document_id"])
                continue
            documents[row["document_id"]] = bundle
            valid_rows.append(row)
        catalog = self.knowledge.catalog(valid_rows)
        analysis = analyze_question(query, catalog) if settings.entity_retrieval else {
            "entities": [], "ambiguous_aliases": [], "queries": [query]}
        if not settings.cross_chapter:
            analysis["queries"] = analysis["queries"][:2]
        if settings.hybrid_search:
            lexical_lists = [("keyword", self.knowledge.search(q, valid_rows, 50)) for q in analysis["queries"]]
            sparse = reciprocal_rank_fusion(lexical_lists, 50)
            matches = reciprocal_rank_fusion([("semantic", result["matches"]), ("keyword", sparse)], 50)
        else:
            matches = result["matches"]
        from local_rag.rag.scenarios import recognize_scene
        if settings.cross_chapter and recognize_scene(query)[0] in {"overview", "timeline", "relationship", "comparison", "cross_chapter"}:
            matches = diversify_chapters(matches, documents, limit)
        result["matches"] = matches[:limit]
        result["book_analysis"] = analysis
        result["book_diagnostics"] = {
            "hybrid_search": settings.hybrid_search, "fusion": "rrf_k60" if settings.hybrid_search else "semantic",
            "lexical_tokenizer": "chinese_bigrams_latin_words", "entity_retrieval": settings.entity_retrieval,
            "cross_chapter": settings.cross_chapter, "books_considered": len(rows),
            "chapters_considered": sum(len(v[2]["sections"]) for v in documents.values()),
            "query_variants": analysis["queries"],
            "resolved_entities": [{"document_id": e["document_id"], "name": e["name"],
                                   "aliases": e["aliases"], "retrieval_aliases": e.get("retrieval_aliases", []),
                                   "alias_candidates": e.get("alias_candidates", []), "matched_names": e["matched_names"]}
                                  for e in analysis["entities"]],
            "ambiguous_aliases": analysis["ambiguous_aliases"], "legacy_fallback_documents": len(fallback),
        }
        return result, documents, catalog

    @timed_operation("retrieval")
    def hybrid_search(self, query, limit=5, document_id=None, settings=None):
        from local_rag.core.config import RagSettings
        try:
            with self.lock:
                result, _, _ = self._book_search(query, limit, document_id, settings or RagSettings.from_env())
                result["strategy"] = result.pop("book_diagnostics")
                result.pop("book_analysis", None)
                return result
        except DocumentError:
            raise
        except Exception as exc:
            logger.exception("Book hybrid retrieval failed")
            raise DocumentError("search_failed", "全书混合检索失败，请检查本地索引与服务日志。", 503) from exc

    @timed_operation("retrieval")
    def retrieve_context(self, query, limit, document_id, plan, settings):
        from local_rag.rag.context import expand_context
        from local_rag.rag.book import knowledge_contexts, combine_contexts
        with self.lock:
            try:
                result, documents, catalog = self._book_search(query, limit, document_id, settings)
                expanded = expand_context(query, result["matches"], documents, plan, settings.context_chunks)
                structured = knowledge_contexts(query, result["book_analysis"], catalog, documents, plan,
                                                settings.context_chunks) if settings.entity_retrieval else []
                for source in structured:
                    source["model_version"] = result["model"]["version"] if result["model"] else None
                result["contexts"] = combine_contexts(structured, expanded, settings.context_chunks, plan.scene)
                if result["contexts"] and result["book_analysis"]["entities"]:
                    result["contexts"][0]["metadata"]["query_people"] = [
                        {"name": e["name"], "queried_as": matched,
                         **({"aliases": e["aliases"]} if plan.scene == "alias" else {}),
                         "kind": "retrieval_hint" if matched in e.get("retrieval_aliases", []) else "explicit_alias"}
                        for e in result["book_analysis"]["entities"] for matched in e["matched_names"] if matched != e["name"] or plan.scene == "alias"]
                result["strategy"] = {**plan.to_dict(), **result.pop("book_diagnostics"),
                                      "structured_contexts": len(structured),
                                      "timeline_order": "document_narrative_order" if plan.scene == "timeline" else None}
                result.pop("book_analysis", None)
                return result
            except DocumentError:
                raise
            except Exception as exc:
                logger.exception("Book context expansion failed")
                raise DocumentError("search_failed", "全书上下文读取失败，请检查文档存储与服务日志。", 503) from exc

    def close(self):
        self._book_cache.clear()
        if getattr(self, "artifacts", None) is not None:
            self.artifacts.close()
        self.vectors.close()
