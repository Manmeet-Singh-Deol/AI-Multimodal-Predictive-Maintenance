"""Embedding generation module for industrial text representations.

Wraps SentenceTransformers (all-MiniLM-L6-v2) with lazy model initialization,
LRU caching for repeated text queries, and batched inference with progress tracking.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from loguru import logger
from tqdm import tqdm

from config.settings import get_settings


class LRUEmbeddingCache:
    """Thread-safe Least Recently Used (LRU) cache for text embeddings.

    Evicts least recently accessed embeddings when capacity is reached,
    preventing duplicate transformer forward passes for identical chunks or queries.
    """

    def __init__(self, capacity: int = 4096) -> None:
        """Initialize the LRU cache with maximum item capacity.

        Args:
            capacity: Maximum number of embeddings to store before evicting.
        """
        self.capacity = max(1, capacity)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _hash_key(self, text: str) -> str:
        """Generate a deterministic SHA-256 hash key for text.

        Args:
            text: Input text string.

        Returns:
            Hex-encoded hash digest.
        """
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Retrieve cached embedding for text if present.

        Args:
            text: Input text string.

        Returns:
            Cached numpy array or None if cache miss.
        """
        key = self._hash_key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache, evicting the oldest if capacity is exceeded.

        Args:
            text: Original input text string.
            embedding: Computed 1D numpy array embedding.
        """
        key = self._hash_key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = embedding
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all stored embeddings from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return the current number of cached embeddings."""
        with self._lock:
            return len(self._cache)


class EmbeddingManager:
    """Generates dense vector embeddings using SentenceTransformers with LRU caching.

    Supports single-text and batched embedding workflows with lazy model loading,
    ensuring zero startup overhead until the first embedding request is made.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_capacity: int = 4096,
        device: str | None = None,
    ) -> None:
        """Initialize EmbeddingManager.

        Args:
            model_name: SentenceTransformers model identifier.
                Defaults to settings.embedding_model ('all-MiniLM-L6-v2').
            cache_capacity: Maximum number of embeddings to retain in LRU cache.
            device: Computing device ('cpu', 'cuda', etc.). Defaults to None (auto-detect).
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self._cache = LRUEmbeddingCache(capacity=cache_capacity)
        self._model: Any = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

        logger.debug(
            f"EmbeddingManager initialized for model '{self.model_name}' "
            f"(cache capacity: {cache_capacity})."
        )

    @property
    def model(self) -> Any:
        """Lazily load and return the underlying SentenceTransformer model.

        Thread-safe singleton pattern ensures the transformer weights are
        loaded into memory only on first demand.

        Returns:
            Loaded SentenceTransformer instance.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(
                        f"Loading SentenceTransformer model '{self.model_name}'..."
                    )
                    try:
                        from sentence_transformers import SentenceTransformer

                        kwargs: dict[str, Any] = {}
                        if self.device:
                            kwargs["device"] = self.device

                        loaded = SentenceTransformer(self.model_name, **kwargs)
                        self._model = loaded
                        self._dimension = loaded.get_sentence_embedding_dimension()
                        logger.info(
                            f"Model '{self.model_name}' loaded successfully "
                            f"(embedding dimension: {self._dimension})."
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to load embedding model '{self.model_name}': {e}"
                        )
                        raise RuntimeError(
                            f"Could not load SentenceTransformer '{self.model_name}': {e}"
                        ) from e
        return self._model

    @property
    def dimension(self) -> int:
        """Return the dimensional size of generated embeddings (e.g., 384).

        Returns:
            Integer embedding dimension.
        """
        if self._dimension is not None:
            return self._dimension
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a dense vector embedding for a single text string.

        Checks the internal LRU cache prior to running transformer inference.

        Args:
            text: Input string to embed.

        Returns:
            1D numpy array representing the normalized embedding vector.
        """
        cleaned = text.strip()
        if not cleaned:
            # Return zero vector for empty strings
            return np.zeros(self.dimension, dtype=np.float32)

        # Cache lookup
        cached = self._cache.get(cleaned)
        if cached is not None:
            return cached

        try:
            raw_emb = self.model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embedding = np.asarray(raw_emb, dtype=np.float32)
            self._cache.put(cleaned, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text with model '{self.model_name}': {e}")
            raise

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Generate normalized vector embeddings for a list of text strings in batches.

        Utilizes the LRU cache for already seen strings and only runs inference
        on uncached items, displaying a progress bar via tqdm.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            List of 1D numpy arrays matching the order of input texts.
        """
        if not texts:
            return []

        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Step 1: Check cache for all texts
        for idx, text in enumerate(texts):
            cleaned = text.strip()
            if not cleaned:
                results[idx] = np.zeros(self.dimension, dtype=np.float32)
                continue

            cached = self._cache.get(cleaned)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(cleaned)

        # Step 2: Compute embeddings for uncached texts in batches
        if uncached_texts:
            logger.debug(
                f"Computing embeddings for {len(uncached_texts)} uncached items "
                f"({len(texts) - len(uncached_texts)} hits in cache)."
            )

            total_uncached = len(uncached_texts)
            progress_bar = tqdm(
                total=total_uncached,
                desc=f"Embedding batch ({self.model_name})",
                unit="chunks",
                leave=False,
            )

            for i in range(0, total_uncached, batch_size):
                chunk_batch = uncached_texts[i : i + batch_size]
                batch_embs = self.model.encode(
                    chunk_batch,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for sub_idx, emb in enumerate(batch_embs):
                    orig_idx = uncached_indices[i + sub_idx]
                    orig_text = uncached_texts[i + sub_idx]
                    arr = np.asarray(emb, dtype=np.float32)
                    results[orig_idx] = arr
                    self._cache.put(orig_text, arr)

                progress_bar.update(len(chunk_batch))

            progress_bar.close()

        # Step 3: Validate and return final array list
        final_embeddings: list[np.ndarray] = []
        for idx, res in enumerate(results):
            if res is None:
                # Fallback safeguard
                logger.warning(f"Embedding at index {idx} was None; substituting zeros.")
                final_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
            else:
                final_embeddings.append(res)

        return final_embeddings

    def clear_cache(self) -> None:
        """Clear all entries from the LRU embedding cache."""
        self._cache.clear()
        logger.debug("LRU embedding cache cleared.")

    @property
    def cache_size(self) -> int:
        """Return the current number of cached embeddings."""
        return len(self._cache)
