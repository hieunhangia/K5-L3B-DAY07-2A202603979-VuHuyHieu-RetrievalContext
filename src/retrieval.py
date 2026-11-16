"""Three distinct retrieval methods for Day 7 - Agentic RAG:
1. Dense Retrieval (Google GenAI embedding + Cosine Similarity)
2. Lexical Retrieval (BM25 with Vietnamese-aware tokenization)
3. Hybrid Retrieval (Dense + BM25 merged via Reciprocal Rank Fusion - RRF)
"""
import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import GOOGLE_EMBEDDING_MODEL, get_google_genai_client

logger = logging.getLogger(__name__)


def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese text into lowercase normalized terms."""
    if not text:
        return []
    # Normalize Unicode NFC
    text = unicodedata.normalize("NFC", text).lower()
    # Normalize punctuation and extra spaces
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


class Retriever(Protocol):
    """Retriever protocol compatible with starter/agentic.py."""

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return list of chunk dicts with id, text, source_id, source, page, metadata, score."""
        ...


class EmbeddingCache:
    """Persistent disk cache for document and query embeddings."""

    def __init__(self, cache_path: Path | str | None = None):
        if cache_path is None:
            self.cache_path = Path("artifacts/embeddings_cache.json")
        else:
            self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        self._load()

    def _hash_key(self, text: str, model: str, dimension: int) -> str:
        key_str = f"{model}:{dimension}:{text}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache from {self.cache_path}: {e}")
                self._cache = {}

    def save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Error saving cache to {self.cache_path}: {e}")

    def get(self, text: str, model: str, dimension: int) -> list[float] | None:
        h = self._hash_key(text, model, dimension)
        return self._cache.get(h)

    def set(self, text: str, model: str, dimension: int, vector: list[float]) -> None:
        h = self._hash_key(text, model, dimension)
        self._cache[h] = vector


def match_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check if chunk satisfies the provided metadata filters."""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for k, v in filters.items():
        if k == "page" and chunk.get("page") != v:
            return False
        if k == "source_id" and chunk.get("source_id") != v:
            return False
        if k in meta and meta[k] != v:
            return False
    return True


# ==============================================================================
# 1. METHOD 1: DENSE RETRIEVAL (Google Embedding + Cosine Similarity)
# ==============================================================================
class DenseRetriever:
    """Dense retriever using Google Gemini embeddings and Cosine Similarity."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_name: str | None = None,
        cache_path: Path | str | None = None,
    ):
        self.chunks = chunks
        self.model_name = model_name or GOOGLE_EMBEDDING_MODEL
        self.cache = EmbeddingCache(cache_path)
        self.client = None
        self.chunk_vectors: np.ndarray | None = None
        self.dimension: int = 3072
        self._ensure_chunk_embeddings()

    def _get_client(self):
        if self.client is None:
            self.client = get_google_genai_client()
        return self.client

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text string with cache lookup and retry."""
        cached = self.cache.get(text, self.model_name, self.dimension)
        if cached is not None:
            return cached

        client = self._get_client()
        for attempt in range(1, 6):
            try:
                res = client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                vec = res.embeddings[0].values
                self.dimension = len(vec)
                self.cache.set(text, self.model_name, self.dimension, vec)
                self.cache.save()
                return vec
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    sleep_time = 3.0 * attempt + 2.0
                    logger.warning(f"Rate limited (429). Retrying in {sleep_time}s (attempt {attempt}/5)...")
                    time.sleep(sleep_time)
                else:
                    raise e
        raise RuntimeError(f"Failed to embed text after 5 attempts: {text[:50]}")

    def _ensure_chunk_embeddings(self) -> None:
        """Ensure all corpus chunks are embedded and indexed."""
        vectors = []
        missing_indices = []
        for idx, chunk in enumerate(self.chunks):
            v = self.cache.get(chunk["text"], self.model_name, self.dimension)
            if v is not None:
                vectors.append(v)
            else:
                vectors.append(None)
                missing_indices.append(idx)

        if missing_indices:
            logger.info(f"Embedding {len(missing_indices)} chunks using {self.model_name}...")
            from concurrent.futures import ThreadPoolExecutor

            def process_idx(idx: int):
                t = self.chunks[idx]["text"]
                vec = self._embed_single(t)
                return idx, vec

            with ThreadPoolExecutor(max_workers=5) as executor:
                for idx, vec in executor.map(process_idx, missing_indices):
                    vectors[idx] = vec

            self.cache.save()

        # Normalize chunk vectors for fast cosine similarity dot product
        mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.chunk_vectors = mat / norms
        logger.info(f"DenseRetriever ready with {len(self.chunks)} chunks (dim: {self.dimension}).")

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        # 1. Embed query
        q_vec = np.array(self._embed_single(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        # 2. Compute Cosine Similarities
        scores = np.dot(self.chunk_vectors, q_vec)

        # 3. Filter and rank
        ranked_indices = np.argsort(scores)[::-1]
        results: list[dict[str, Any]] = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if not match_filters(chunk, filters):
                continue
            item = dict(chunk)
            item["score"] = float(scores[idx])
            results.append(item)
            if len(results) >= top_k:
                break

        return results


# ==============================================================================
# 2. METHOD 2: LEXICAL RETRIEVAL (BM25 with Vietnamese Tokenization)
# ==============================================================================
class BM25Retriever:
    """Lexical retriever using BM25Okapi and Vietnamese text normalization."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize_vi(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"BM25Retriever initialized with {len(chunks)} chunks.")

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        q_tokens = tokenize_vi(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if not match_filters(chunk, filters):
                continue
            item = dict(chunk)
            item["score"] = float(scores[idx])
            results.append(item)
            if len(results) >= top_k:
                break

        return results


# ==============================================================================
# 3. METHOD 3: HYBRID RETRIEVAL (Dense + BM25 fused with RRF)
# ==============================================================================
class HybridRetriever:
    """Hybrid retriever combining Dense and Lexical BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        pool_size: int = 20,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        logger.info(
            f"HybridRetriever initialized (rrf_k={rrf_k}, pool_size={pool_size}, "
            f"dense_weight={dense_weight}, bm25_weight={bm25_weight})."
        )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        # 1. Retrieve candidates from both systems
        dense_results = self.dense.search(query, filters=filters, top_k=self.pool_size)
        bm25_results = self.bm25.search(query, filters=filters, top_k=self.pool_size)

        # 2. Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum(1 / (k + rank(d)))
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, chunk in enumerate(dense_results, 1):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, chunk in enumerate(bm25_results, 1):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        # 3. Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results: list[dict[str, Any]] = []
        for cid in sorted_cids[:top_k]:
            item = dict(chunk_map[cid])
            item["score"] = float(rrf_scores[cid])
            results.append(item)

        return results
