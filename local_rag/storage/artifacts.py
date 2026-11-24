"""Normalized SQLite artifact records, field assertions, and index visibility."""

from contextlib import contextmanager
import json
import sqlite3
from uuid import uuid4

from local_rag.core.exceptions import DocumentError
from local_rag.storage.registry import now


class ArtifactRegistry:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"Unsupported artifact schema version: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    content_hash TEXT NOT NULL, name TEXT NOT NULL, era TEXT, material TEXT,
                    purpose TEXT, cultural_value TEXT, museum TEXT, accession_number TEXT,
                    index_status TEXT NOT NULL DEFAULT 'pending', index_error TEXT,
                    collection_name TEXT, generation TEXT, indexed_revision INTEGER,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS artifact_content
                    ON artifacts(content_hash) WHERE index_status != 'deleted';
                CREATE UNIQUE INDEX IF NOT EXISTS artifact_accession
                    ON artifacts(museum, accession_number)
                    WHERE museum IS NOT NULL AND accession_number IS NOT NULL AND index_status != 'deleted';
                CREATE INDEX IF NOT EXISTS artifact_museum ON artifacts(museum);
                CREATE INDEX IF NOT EXISTS artifact_name ON artifacts(name);
                CREATE TABLE IF NOT EXISTS artifact_claims (
                    id INTEGER PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                    field TEXT NOT NULL, value TEXT NOT NULL, document_id TEXT NOT NULL,
                    source_type TEXT NOT NULL, quote TEXT NOT NULL, filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL, start_index INTEGER NOT NULL, end_index INTEGER NOT NULL,
                    page_numbers TEXT NOT NULL, chunk_ids TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS claim_artifact ON artifact_claims(artifact_id);
                CREATE INDEX IF NOT EXISTS claim_document ON artifact_claims(document_id);
                PRAGMA user_version = 1;
            """)
            db.execute("UPDATE artifacts SET index_status='pending', generation=NULL, "
                       "index_error='上次索引中断，请重建索引。' WHERE index_status='indexing'")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM artifacts WHERE index_status != 'deleted' ORDER BY created_at, artifact_id")]

    def get(self, artifact_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM artifacts WHERE artifact_id=? AND index_status != 'deleted'", (artifact_id,)).fetchone()
            if not row:
                raise DocumentError("artifact_not_found", "文物档案不存在。", 404)
            return dict(row)

    def claims(self, artifact_id):
        with self.connection() as db:
            claims = [dict(r) for r in db.execute("SELECT * FROM artifact_claims WHERE artifact_id=? ORDER BY id", (artifact_id,))]
        for claim in claims:
            claim.pop("id")
            claim.pop("artifact_id")
            for key in ("page_numbers", "chunk_ids"):
                claim[key] = json.loads(claim[key])
        return claims

    def save(self, claims, fields, digest, artifact_id=None, expected_revision=None):
        stamp = now()
        try:
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                if artifact_id is None:
                    existing = db.execute("SELECT artifact_id FROM artifacts WHERE content_hash=? AND index_status != 'deleted'", (digest,)).fetchone()
                    if existing:
                        return existing[0], False
                    artifact_id = "WW-" + uuid4().hex.upper()
                    db.execute("INSERT INTO artifacts (artifact_id, revision, content_hash, name, created_at, updated_at) VALUES (?, 1, ?, ?, ?, ?)",
                               (artifact_id, digest, fields["name"], stamp, stamp))
                else:
                    row = db.execute("SELECT revision FROM artifacts WHERE artifact_id=? AND index_status != 'deleted'", (artifact_id,)).fetchone()
                    if not row:
                        raise DocumentError("artifact_not_found", "文物档案不存在。", 404)
                    if expected_revision != row[0]:
                        raise DocumentError("artifact_revision_conflict", "档案已更新，或缺少 expected_revision；请刷新后再保存。", 409)
                    db.execute("UPDATE artifacts SET revision=revision+1 WHERE artifact_id=?", (artifact_id,))
                    db.execute("DELETE FROM artifact_claims WHERE artifact_id=?", (artifact_id,))
                columns = ("name", "era", "material", "purpose", "cultural_value", "museum", "accession_number")
                db.execute("UPDATE artifacts SET " + ", ".join(f"{c}=?" for c in columns)
                           + ", content_hash=?, updated_at=?, index_status='pending', index_error=NULL, generation=NULL, indexed_revision=NULL WHERE artifact_id=?",
                           (*[fields.get(c) for c in columns], digest, stamp, artifact_id))
                for c in claims:
                    db.execute("""INSERT INTO artifact_claims
                        (artifact_id, field, value, document_id, source_type, quote, filename, file_hash, start_index, end_index, page_numbers, chunk_ids)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                               (artifact_id, c["field"], c["value"], c["document_id"], c["source_type"], c["quote"], c["filename"], c["file_hash"],
                                c["start_index"], c["end_index"], json.dumps(c["page_numbers"]), json.dumps(c["chunk_ids"])))
            return artifact_id, True
        except sqlite3.IntegrityError as exc:
            raise DocumentError("artifact_duplicate", "相同档案或同馆同编号文物已存在，请更新已有档案。", 409) from exc

    def index_state(self, artifact_id, status, *, error=None, collection=None, generation=None, revision=None):
        with self.connection() as db:
            db.execute("""UPDATE artifacts SET index_status=?, index_error=?, collection_name=?, generation=?, indexed_revision=?
                WHERE artifact_id=?""", (status, error, collection, generation, revision, artifact_id))

    def referencing(self, document_id):
        with self.connection() as db:
            return [r[0] for r in db.execute("SELECT DISTINCT artifact_id FROM artifact_claims WHERE document_id=?", (document_id,))]
