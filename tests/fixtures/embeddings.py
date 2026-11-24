"""Deterministic test vectors only: these do not claim BGE-M3 semantic quality."""

import hashlib
import re

from local_rag.embeddings.bge import ModelIdentity


class FakeEmbeddings:
    def __init__(self):
        self.version = "synthetic-test-v1"
        self.calls = 0

    def identity(self):
        return ModelIdentity("bge-m3:latest", self.version)

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self.embed([text])[0]

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * 1024
            for token in re.findall(r"\w+", text.lower()):
                slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 1024
                vector[slot] += 1
            if not any(vector):
                vector[0] = 1
            vectors.append(vector)
        return vectors
