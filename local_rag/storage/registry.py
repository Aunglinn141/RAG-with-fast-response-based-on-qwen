"""SQLite is the authoritative source for index visibility and processing state."""

from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3


def now():
    return datetime.now(timezone.utc).isoformat()


class DocumentRegistry:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"Unsupported document registry schema version: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    model_name TEXT,
                    model_version TEXT,
                    embedding_dimension INTEGER,
                    collection_name TEXT,
                    generation TEXT,
                    indexed_at TEXT,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS documents_status ON documents(status);
                CREATE INDEX IF NOT EXISTS documents_hash ON documents(file_hash);
                PRAGMA user_version = 1;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, record):
        with self.connection() as db:
            db.execute("""INSERT OR IGNORE INTO documents
                (document_id, filename, file_hash, uploaded_at, status, chunk_count, updated_at)
                VALUES (?, ?, ?, ?, 'ready', ?, ?)""",
                (record["document_id"], record["filename"], record["sha256"],
                 record["created_at"], record["chunk_count"], now()))

    def get(self, document_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            return dict(row) if row else None

    def rows(self, *, include_deleted=False):
        with self.connection() as db:
            sql = "SELECT * FROM documents"
            if not include_deleted:
                sql += " WHERE status != 'deleted'"
            return [dict(row) for row in db.execute(sql + " ORDER BY uploaded_at, document_id")]

    def update(self, document_id, **values):
        allowed = {"status", "chunk_count", "model_name", "model_version", "embedding_dimension",
                   "collection_name", "generation", "indexed_at", "last_error"}
        if not values or values.keys() - allowed:
            raise ValueError("Invalid registry update")
        values["updated_at"] = now()
        with self.connection() as db:
            db.execute("UPDATE documents SET " + ", ".join(f"{key} = ?" for key in values)
                       + " WHERE document_id = ?", (*values.values(), document_id))
