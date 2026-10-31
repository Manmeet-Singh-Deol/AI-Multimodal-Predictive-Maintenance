"""Hybrid retriever combining dense semantic search and sparse BM25 keyword search.

Implements Reciprocal Rank Fusion (RRF) to merge semantic similarity results from
ChromaDB with exact keyword matching from BM25, optimizing retrieval accuracy for
both technical jargon/part numbers and semantic natural language troubleshooting queries.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from loguru import logger

from config.settings import get_settings
from .document_processor import Document
from .vector_store import VectorStoreManager

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class HybridRetriever:
    """Combines dense semantic vector search with sparse BM25 keyword retrieval.

    Utilizes Reciprocal Rank Fusion (RRF) parameterized by alpha:
        alpha = 1.0 -> purely semantic vector search
        alpha = 0.0 -> purely keyword BM25 search
        alpha = 0.7 (default) -> balanced hybrid favoring semantics with keyword precision
    """

    # RRF smoothing constant (Cormack et al. 2009 standard)
    K_RRF = 60

    def __init__(
        self,
        vector_store: VectorStoreManager,
        documents: list[Document] | None = None,
        alpha: float | None = None,
    ) -> None:
        """Initialize HybridRetriever with vector store and BM25 index.

        Args:
            vector_store: Instantiated VectorStoreManager for semantic retrieval.
            documents: Optional list of Document objects to initialize BM25.
                If None, loads existing documents from vector_store.
            alpha: Default hybrid weight (1.0 = semantic, 0.0 = keyword).
                Defaults to settings.hybrid_search_alpha (0.7).
        """
        settings = get_settings()
        self.vector_store = vector_store
        self.alpha = alpha if alpha is not None else settings.hybrid_search_alpha

        if BM25Okapi is None:
            logger.warning(
                "rank_bm25 is not installed. Hybrid search will fallback to semantic search only. "
                "Install via `pip install rank-bm25`."
            )

        self.documents: list[Document] = []
        self.bm25: Any = None
        self.build_bm25_index(documents)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase terms, preserving alphanumeric codes and hyphens.

        Important for industrial part numbers, error codes, and machine specs
        (e.g., 'E-401', 'ISO_VG46', '3000RPM').

        Args:
            text: Raw input text.

        Returns:
            List of normalized string tokens.
        """
        cleaned = text.lower()
        # Extract word characters, hyphens, and underscores
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", cleaned)
        return [t for t in tokens if len(t) > 1]

    def build_bm25_index(self, documents: list[Document] | None = None) -> None:
        """Build or refresh the BM25 index from a document list or vector store.

        Args:
            documents: Optional document list. If None, queries vector_store.
        """
        if BM25Okapi is None:
            return

        if documents is not None:
            self.documents = list(documents)
        else:
            self.documents = self.vector_store.get_all_documents()

        if not self.documents:
            logger.info("No documents available to build BM25 index yet.")
            self.bm25 = None
            return

        logger.info(f"Building BM25 keyword index over {len(self.documents)} chunks...")
        tokenized_corpus = [self.tokenize(doc.content) for doc in self.documents]

        try:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index constructed successfully.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents using dense vector similarity.

        Args:
            query: User search query.
            top_k: Number of top documents to return.
            metadata_filter: Optional ChromaDB metadata filter.

        Returns:
            List of (Document, score) tuples.
        """
        return self.vector_store.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents using sparse BM25 keyword matching.

        Args:
            query: User search query.
            top_k: Number of top documents to return.

        Returns:
            List of (Document, score) tuples normalized to [0, 1].
        """
        if self.bm25 is None or not self.documents:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        try:
            scores = self.bm25.get_scores(tokens)
        except Exception as e:
            logger.error(f"BM25 query scoring failed: {e}")
            return []

        if len(scores) == 0:
            return []

        # Find top indices with non-zero scores
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0.0]
        if not indexed_scores:
            return []

        # Sort descending by score
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = indexed_scores[:top_k]

        max_score = top_matches[0][1] if top_matches else 1.0
        results: list[tuple[Document, float]] = []

        for idx, score in top_matches:
            normalized_score = score / max_score if max_score > 0 else 0.0
            results.append((self.documents[idx], round(normalized_score, 4)))

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform hybrid retrieval using Reciprocal Rank Fusion (RRF).

        Combines semantic similarity and BM25 keyword rankings according to alpha.
        RRF formula:
            RRF_score(d) = alpha * (1 / (K + rank_sem)) + (1 - alpha) * (1 / (K + rank_bm25))

        Args:
            query: Input natural language question or keywords.
            top_k: Number of top results to return.
            alpha: Weight balancing semantic vs keyword search (0.0 to 1.0).
                If None, uses self.alpha.
            metadata_filter: Optional metadata filter for vector search.

        Returns:
            List of (Document, score) tuples sorted by fused score descending.
        """
        effective_alpha = self.alpha if alpha is None else float(alpha)
        effective_alpha = max(0.0, min(1.0, effective_alpha))

        # Fallback to purely semantic if BM25 is not available or alpha == 1.0
        if self.bm25 is None or effective_alpha == 1.0:
            return self.semantic_search(query, top_k=top_k, metadata_filter=metadata_filter)

        # Fallback to purely keyword if alpha == 0.0
        if effective_alpha == 0.0:
            return self.keyword_search(query, top_k=top_k)

        # Retrieve a candidate pool from both retrievers
        candidate_pool_size = max(top_k * 3, 20)

        semantic_results = self.semantic_search(
            query=query,
            top_k=candidate_pool_size,
            metadata_filter=metadata_filter,
        )
        keyword_results = self.keyword_search(
            query=query,
            top_k=candidate_pool_size,
        )

        if not semantic_results and not keyword_results:
            return []
        if not semantic_results:
            return keyword_results[:top_k]
        if not keyword_results:
            return semantic_results[:top_k]

        # Deduplicate and compute RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_registry: dict[str, Document] = {}

        def get_doc_unique_key(doc: Document) -> str:
            source = str(doc.metadata.get("source", "unknown"))
            page = str(doc.metadata.get("page", "1"))
            chunk_id = str(doc.metadata.get("chunk_id", "0"))
            return f"{source}::p{page}::c{chunk_id}"

        # 1. Semantic ranks
        for rank, (doc, _sim) in enumerate(semantic_results, start=1):
            key = get_doc_unique_key(doc)
            rrf_scores[key] += effective_alpha * (1.0 / (self.K_RRF + rank))
            if key not in doc_registry:
                doc_registry[key] = doc

        # 2. BM25 ranks
        for rank, (doc, _kw_score) in enumerate(keyword_results, start=1):
            key = get_doc_unique_key(doc)
            rrf_scores[key] += (1.0 - effective_alpha) * (1.0 / (self.K_RRF + rank))
            if key not in doc_registry:
                doc_registry[key] = doc

        # Theoretical maximum score (rank 1 in both):
        max_possible_rrf = (effective_alpha / (self.K_RRF + 1)) + (
            (1.0 - effective_alpha) / (self.K_RRF + 1)
        )

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )

        final_results: list[tuple[Document, float]] = []
        for key in sorted_keys[:top_k]:
            normalized_score = rrf_scores[key] / max_possible_rrf if max_possible_rrf > 0 else 0.0
            final_results.append((doc_registry[key], round(min(1.0, normalized_score), 4)))

        return final_results

    @staticmethod
    def format_sources(
        results: list[tuple[Document, float]],
    ) -> list[dict[str, Any]]:
        """Convert retrieval results into structured citation records.

        Args:
            results: List of (Document, score) tuples.

        Returns:
            List of dictionaries with source, page, section, and score.
        """
        citations: list[dict[str, Any]] = []
        for doc, score in results:
            meta = doc.metadata
            citations.append(
                {
                    "source": meta.get("source", "Unknown Manual"),
                    "page": meta.get("page", 1),
                    "section": meta.get("section_title", "General"),
                    "chunk_id": meta.get("chunk_id", 0),
                    "relevance_score": float(score),
                }
            )
        return citations

    @staticmethod
    def format_retrieval_context(
        results: list[tuple[Document, float]],
    ) -> str:
        """Format retrieved chunks into a clean, markdown context block for LLM prompts.

        Args:
            results: List of (Document, score) tuples.

        Returns:
            Formatted string containing source citations and text excerpts.
        """
        if not results:
            return "No relevant maintenance manual sections found."

        context_blocks: list[str] = []
        for idx, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata
            source = meta.get("source", "Manual")
            page = meta.get("page", "N/A")
            section = meta.get("section_title", "General")

            header = (
                f"--- [DOCUMENT EXCERPT {idx}] ---\n"
                f"Source: {source} | Page: {page} | Section: {section} | Relevance: {score:.2f}"
            )
            body = doc.content.strip()
            context_blocks.append(f"{header}\n{body}")

        return "\n\n".join(context_blocks)
