"""Persistent Chroma storage with explicitly supplied vectors and model isolation."""

import hashlib
import json

import chromadb
from chromadb.config import Settings


class ChromaStore:
    PREFIX = "localrag_chunks_"

    def __init__(self, path):
        self.path = path
        self._client = None
        self._collections = {}

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=Settings(anonymized_telemetry=False))
        return self._client

    @classmethod
    def collection_name(cls, identity):
        key = f"{identity.name}\n{identity.version}\n{identity.dimension}"
        return cls.PREFIX + hashlib.sha256(key.encode()).hexdigest()[:32]

    def collection(self, name):
        if name not in self._collections:
            self._collections[name] = self.client.get_collection(name, embedding_function=None)
        return self._collections[name]

    def prepare(self, identity):
        name = self.collection_name(identity)
        self._collections[name] = self.client.get_or_create_collection(
            name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
            metadata={"model_name": identity.name, "model_version": identity.version,
                      "embedding_dimension": identity.dimension})
        return name

    def upsert(self, name, document_id, generation, chunks, vectors):
        metadata = []
        for chunk in chunks:
            source = chunk["metadata"]
            entry = {key: value for key, value in source.items()
                     if type(value) in (str, int, float, bool)}
            entry.update(document_id=document_id, generation=generation, chunk_id=chunk["chunk_id"],
                         source_metadata=json.dumps(source, ensure_ascii=False))
            metadata.append(entry)
        self.collection(name).upsert(
            ids=[f"{document_id}:{generation}:{c['metadata']['chunk_index']}" for c in chunks],
            documents=[c["text"] for c in chunks], metadatas=metadata, embeddings=vectors)

    def count_document(self, document_id):
        return sum(len(collection.get(where={"document_id": document_id}, include=[])["ids"])
                   for collection in self.managed_collections())

    def managed_collections(self):
        # Do not create a vector database just to remove a never-indexed document.
        if self._client is None and not (self.path / "chroma.sqlite3").exists():
            return []
        return [self.collection(c.name) for c in self.client.list_collections()
                if c.name.startswith(self.PREFIX)]

    def delete_document(self, document_id):
        for collection in self.managed_collections():
            collection.delete(where={"document_id": document_id})
        if self.count_document(document_id):
            raise RuntimeError("Chroma still contains vectors for this document")

    def search(self, name, vector, rows, limit):
        # Both filters exclude obsolete generations and interrupted operations.
        where = {"$and": [
            {"document_id": {"$in": [r["document_id"] for r in rows]}},
            {"generation": {"$in": [r["generation"] for r in rows]}},
        ]}
        collection = self.collection(name)
        count = collection.count()
        if not count:
            return []
        result = collection.query(query_embeddings=[vector], where=where,
                                  n_results=min(limit, count), include=["documents", "metadatas", "distances"])
        matches = []
        for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            matches.append({"chunk_id": metadata["chunk_id"], "text": text,
                            "metadata": json.loads(metadata["source_metadata"]),
                            "distance": distance, "score": 1 - distance})
        return matches

    def close(self):
        self._collections.clear()
        if self._client is not None:
            self._client.close()
            self._client = None
