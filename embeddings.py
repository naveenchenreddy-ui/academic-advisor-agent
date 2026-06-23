"""
Simple deterministic embeddings using char n-grams.
Replace SimpleHashEmbeddings with OpenAIEmbeddings or
HuggingFaceEmbeddings in production for better semantic search.
"""
import hashlib
import math
from typing import List
from langchain.embeddings.base import Embeddings

EMBED_DIMENSION = 1536


class SimpleHashEmbeddings(Embeddings):
    """Deterministic 1536-d char trigram embeddings. No API key needed."""

    def __init__(self, dim: int = EMBED_DIMENSION):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        text = text.lower()[:2000]
        vec = [0.0] * self.dim
        for i in range(len(text) - 2):
            h = int(hashlib.md5(text[i:i+3].encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)
