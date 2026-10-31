"""RAG engine for retrieval-augmented generation over maintenance manuals.

Provides complete document processing, dense vector embeddings, persistent ChromaDB
vector storage, hybrid BM25 + semantic retrieval with Reciprocal Rank Fusion,
and Gemini-powered generative troubleshooting synthesis with multimodal vision.
"""

from .document_processor import Document, DocumentProcessor
from .embeddings import EmbeddingManager, LRUEmbeddingCache
from .rag_chain import RAGChain, RAGResponse
from .retriever import HybridRetriever
from .vector_store import VectorStoreManager

__all__ = [
    "Document",
    "DocumentProcessor",
    "EmbeddingManager",
    "LRUEmbeddingCache",
    "VectorStoreManager",
    "HybridRetriever",
    "RAGChain",
    "RAGResponse",
]
