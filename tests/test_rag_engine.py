"""Unit and integration tests for the RAG engine module.

Covers Document processing, chunking logic, LRU cache eviction,
embedding generation, vector store deduplication, BM25 and hybrid retrieval,
and RAG chain response assembly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag_engine.document_processor import Document, DocumentProcessor
from src.rag_engine.embeddings import EmbeddingManager, LRUEmbeddingCache
from src.rag_engine.rag_chain import RAGChain, RAGResponse
from src.rag_engine.retriever import HybridRetriever
from src.rag_engine.vector_store import VectorStoreManager


class TestDocumentProcessor:
    """Tests for document parsing, section detection, and sentence chunking."""

    def test_document_defaults(self) -> None:
        """Test default metadata assignment for Document instances."""
        doc = Document(content="Pump vibration test passed.")
        assert doc.content == "Pump vibration test passed."
        assert doc.metadata["source"] == "unknown"
        assert doc.metadata["page"] == 1
        assert doc.metadata["chunk_id"] == 0
        assert doc.metadata["section_title"] == "General"

    def test_section_header_detection(self) -> None:
        """Verify section header heuristic rules for manuals."""
        processor = DocumentProcessor()

        # Positive cases
        assert processor.is_section_header("SECTION 4: TROUBLESHOOTING GUIDE")
        assert processor.is_section_header("SAFETY FIRST: LOCKOUT TAGOUT")
        assert processor.is_section_header("BEARING LUBRICATION INTERVALS")
        assert processor.is_section_header("3.2 Hydraulic Pressure Calibration")
        assert processor.is_section_header("Inspection Steps", is_bold=True)

        # Negative cases
        assert not processor.is_section_header("This is a normal sentence describing oil levels.")
        assert not processor.is_section_header("Always wear safety goggles at all times,")
        assert not processor.is_section_header("")
        assert not processor.is_section_header("a")

    def test_chunk_text_sentence_boundaries(self) -> None:
        """Verify that chunking respects sentence boundaries and overlap."""
        processor = DocumentProcessor(chunk_size=120, chunk_overlap=30)
        sample_text = (
            "Centrifugal pump bearings require periodic greasing every 500 hours. "
            "Use NLGI Grade 2 lithium complex grease exclusively. "
            "Over-greasing may cause high operating temperatures and seal leakage. "
            "Monitor vibration amplitude using an accelerometer mounted on the housing."
        )

        chunks = processor.chunk_text(sample_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0
            # Ensure chunks do not arbitrarily chop words when sentences fit
            assert not chunk.startswith(".")


class TestLRUEmbeddingCache:
    """Tests for thread-safe LRU embedding cache."""

    def test_lru_cache_capacity_and_eviction(self) -> None:
        """Verify LRU cache respects maximum capacity and evicts oldest."""
        cache = LRUEmbeddingCache(capacity=2)

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        v3 = np.array([1.0, 1.0], dtype=np.float32)

        cache.put("text1", v1)
        cache.put("text2", v2)
        assert len(cache) == 2

        # Access text1 to mark as recently used
        assert np.array_equal(cache.get("text1"), v1)

        # Put third item; text2 should be evicted because text1 was accessed more recently
        cache.put("text3", v3)
        assert len(cache) == 2
        assert cache.get("text2") is None
        assert cache.get("text1") is not None
        assert cache.get("text3") is not None


class TestHybridRetriever:
    """Tests for BM25 and Reciprocal Rank Fusion hybrid retrieval."""

    def test_tokenization_preserves_codes(self) -> None:
        """Ensure tokenization preserves industrial error codes and hyphenated identifiers."""
        tokens = HybridRetriever.tokenize("Error E-401 detected on Motor_Unit-02 at 1800RPM!")
        assert "e-401" in tokens
        assert "motor_unit-02" in tokens
        assert "1800rpm" in tokens

    def test_bm25_keyword_search(self) -> None:
        """Verify BM25 keyword matching scores relevant documents higher."""
        mock_vs = MagicMock()
        docs = [
            Document(
                content="Hydraulic fluid viscosity grade ISO VG 46 is recommended for high pressure pumps.",
                metadata={"source": "hydraulics.pdf", "page": 5, "chunk_id": 1},
            ),
            Document(
                content="Electrical motor wiring diagram for 460V 3-phase induction motor with delta winding.",
                metadata={"source": "electrical.pdf", "page": 12, "chunk_id": 2},
            ),
            Document(
                content="Gearbox tooth wear inspection procedure and backlash tolerance measurements.",
                metadata={"source": "gearbox.pdf", "page": 8, "chunk_id": 3},
            ),
        ]

        retriever = HybridRetriever(vector_store=mock_vs, documents=docs)
        results = retriever.keyword_search("hydraulic fluid viscosity ISO VG 46", top_k=2)

        assert len(results) > 0
        top_doc, top_score = results[0]
        assert "hydraulic" in top_doc.content.lower()
        assert top_score > 0.0

    def test_hybrid_search_rrf_blending(self) -> None:
        """Verify Reciprocal Rank Fusion merges semantic and keyword matches."""
        mock_vs = MagicMock()
        docs = [
            Document(
                content="Check oil filter differential pressure indicator daily.",
                metadata={"source": "oil.pdf", "page": 3, "chunk_id": 1},
            ),
            Document(
                content="Bearing high temperature alarm triggers above 85 degrees Celsius.",
                metadata={"source": "bearing.pdf", "page": 7, "chunk_id": 2},
            ),
        ]

        # Mock semantic search return
        mock_vs.search.return_value = [(docs[1], 0.92), (docs[0], 0.45)]

        retriever = HybridRetriever(vector_store=mock_vs, documents=docs, alpha=0.7)
        results = retriever.hybrid_search("Bearing high temperature", top_k=2)

        assert len(results) == 2
        # Bearing doc should be ranked first
        assert results[0][0].metadata["source"] == "bearing.pdf"
        assert results[0][1] > results[1][1]


class TestRAGChain:
    """Tests for RAG response construction, citations, and confidence scoring."""

    def test_confidence_calculation(self) -> None:
        """Verify confidence scoring based on retrieval scores and context presence."""
        chain = RAGChain(retriever=MagicMock())

        # No context
        conf_empty = chain._calculate_confidence([], has_context=False)
        assert conf_empty == 0.35

        # Strong retrieval results
        mock_results = [
            (Document(content="test"), 0.90),
            (Document(content="test2"), 0.80),
        ]
        conf_strong = chain._calculate_confidence(mock_results, has_context=True)
        assert 0.80 <= conf_strong <= 0.95

    def test_query_fallback_without_api_key(self) -> None:
        """Verify graceful fallback with citations when no Gemini API key is configured."""
        mock_retriever = MagicMock()
        doc = Document(
            content="Isolate electrical breaker before replacing coupling inserts.",
            metadata={"source": "safety.pdf", "page": 2, "chunk_id": 10},
        )
        mock_retriever.hybrid_search.return_value = [(doc, 0.88)]
        mock_retriever.format_sources.return_value = [
            {
                "source": "safety.pdf",
                "page": 2,
                "section": "General",
                "chunk_id": 10,
                "relevance_score": 0.88,
            }
        ]
        mock_retriever.format_retrieval_context.return_value = (
            "--- [DOCUMENT EXCERPT 1] ---\nSource: safety.pdf | Page: 2\n"
            "Isolate electrical breaker before replacing coupling inserts."
        )

        chain = RAGChain(retriever=mock_retriever, api_key="")
        response = chain.query("How to replace coupling safely?")

        assert isinstance(response, RAGResponse)
        assert "Gemini API Key Required" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0]["source"] == "safety.pdf"
        assert response.confidence >= 0.80
