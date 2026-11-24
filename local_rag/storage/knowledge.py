"""Persistent derived book index: Chinese bigram BM25, aliases, relation/event spans."""
from contextlib import contextmanager
import json
import re
import sqlite3

from local_rag.documents.hierarchy import STOP
from local_rag.documents.knowledge import VERSION, build_knowledge, knowledge_fingerprint


def tokens(text):
    result = []
    for word in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            result.extend(word[i:i + 2] for i in range(len(word) - 1) if word[i:i + 2] not in STOP)
        elif len(word) > 1:
            result.append(word)
    return result


class KnowledgeStore:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS books (document_id TEXT PRIMARY KEY, generation TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS book_chunks USING fts5(tokens, document_id UNINDEXED, chunk_id UNINDEXED, payload UNINDEXED)")
        self.cache = {}

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def delete(self, document_id):
        with self.connect() as db:
            db.execute("DELETE FROM book_chunks WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM books WHERE document_id=?", (document_id,))
        self.cache.pop(document_id, None)

    def sync(self, row, record, text, hierarchy):
        key = row["document_id"]
        fingerprint = knowledge_fingerprint(record, text, hierarchy)
        with self.connect() as db:
            old = db.execute("SELECT generation,fingerprint FROM books WHERE document_id=?", (key,)).fetchone()
            if old == (row["generation"], fingerprint):
                return
            payload = build_knowledge(record, text, hierarchy)
            db.execute("DELETE FROM book_chunks WHERE document_id=?", (key,))
            db.executemany("INSERT INTO book_chunks(tokens,document_id,chunk_id,payload) VALUES (?,?,?,?)",
                           [(" ".join(tokens(c["text"])), key, c["chunk_id"], json.dumps(c, ensure_ascii=False))
                            for c in record["chunks"]])
            db.execute("INSERT OR REPLACE INTO books VALUES (?,?,?,?)",
                       (key, row["generation"], fingerprint, json.dumps(payload, ensure_ascii=False)))
        self.cache.pop(key, None)

    def catalog(self, rows):
        result = {}
        with self.connect() as db:
            for row in rows:
                stored = db.execute("SELECT fingerprint FROM books WHERE document_id=? AND generation=?",
                                    (row["document_id"], row["generation"])).fetchone()
                if not stored:
                    continue
                key = (row["document_id"], row["generation"], stored[0])
                cached = self.cache.get(row["document_id"])
                if cached is None or cached[0] != key:
                    payload = db.execute("SELECT payload FROM books WHERE document_id=? AND generation=? AND fingerprint=?",
                                         (row["document_id"], row["generation"], stored[0])).fetchone()
                    if payload is None:
                        continue
                    cached = (key, json.loads(payload[0]))
                    # Cache at most four books; persistence remains authoritative.
                    if len(self.cache) >= 4:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[row["document_id"]] = cached
                result[row["document_id"]] = cached[1]
        return result

    def search(self, question, rows, limit=50):
        words = sorted(set(tokens(question)))[:96]
        if not words or not rows:
            return []
        expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
        visible = {row["document_id"]: row for row in rows}
        placeholders = ",".join("?" for _ in visible)
        # Filter before LIMIT: deleted/incompatible documents cannot crowd out visible hits.
        sql = ("SELECT payload, bm25(book_chunks) FROM book_chunks WHERE book_chunks MATCH ? "
               f"AND document_id IN ({placeholders}) ORDER BY bm25(book_chunks),chunk_id LIMIT ?")
        with self.connect() as db:
            matches = db.execute(sql, (expression, *visible, limit)).fetchall()
        result = []
        for payload, rank in matches:
            chunk = json.loads(payload)
            row = visible[chunk["metadata"]["document_id"]]
            result.append({**chunk, "score": 0.0, "distance": 1.0, "model_version": row["model_version"],
                           "keyword_bm25": -rank})
        return result
