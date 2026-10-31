"""ChromaDB vector store manager for maintenance documentation.

Provides persistent storage, vector indexing, metadata filtering,
deduplication by source and chunk ID, and similarity search for technical manuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings
from .document_processor import Document
from .embeddings import EmbeddingManager

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage for maintenance manuals.

    Handles document ingestion with deduplication, embedding calculation,
    similarity search with metadata filtering, and collection lifecycle management.
    """

    def __init__(
        self,
        collection_name: str = "maintenance_docs",
        persist_dir: str | Path | None = None,
        embedding_manager: EmbeddingManager | None = None,
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection. Defaults to 'maintenance_docs'.
            persist_dir: Local path for persistent database storage.
                Defaults to settings.chroma_persist_dir.
            embedding_manager: EmbeddingManager instance for vector generation.
                If None, instantiates a default EmbeddingManager.

        Raises:
            RuntimeError: If chromadb package is not installed.
        """
        if chromadb is None:
            raise RuntimeError(
                "chromadb package is required. Install via `pip install chromadb`."
            )

        settings = get_settings()
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_manager = embedding_manager or EmbeddingManager()

        logger.info(
            f"Initializing ChromaDB PersistentClient at '{self.persist_dir}' "
            f"(collection: '{self.collection_name}')."
        )

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True)
                if ChromaSettings
                else None,
            )
            # Use cosine distance space
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' ready with "
                f"{self.collection.count()} existing documents."
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}")
            raise RuntimeError(f"ChromaDB initialization failure: {e}") from e

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """Sanitize metadata for ChromaDB compatibility.

        ChromaDB only supports primitive types: str, int, float, and bool.
        Complex objects or None values are converted to strings or skipped.

        Args:
            meta: Raw dictionary of metadata.

        Returns:
            Sanitized dictionary containing only supported types.
        """
        sanitized: dict[str, str | int | float | bool] = {}
        for key, val in meta.items():
            if val is None:
                continue
            if isinstance(val, (str, int, float, bool)):
                sanitized[key] = val
            else:
                sanitized[key] = str(val)
        return sanitized

    @staticmethod
    def generate_doc_id(doc: Document, fallback_idx: int = 0) -> str:
        """Create a deterministic unique identifier for a Document chunk.

        Combines source filename and chunk_id to enforce deduplication.

        Args:
            doc: Document instance.
            fallback_idx: Fallback integer if chunk_id is not specified.

        Returns:
            Unique string identifier.
        """
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        chunk_id = doc.metadata.get("chunk_id", fallback_idx)
        clean_source = source.replace(" ", "_")
        return f"{clean_source}::chunk_{chunk_id}"

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 64,
    ) -> list[str]:
        """Embed and store documents in the ChromaDB collection.

        Deduplicates against existing items: skips any document whose source and
        chunk_id combination already exists in the collection.

        Args:
            documents: List of Document objects to ingest.
            batch_size: Batch size for embedding and writing.

        Returns:
            List of newly added document IDs.
        """
        if not documents:
            logger.debug("add_documents called with an empty list; nothing to add.")
            return []

        # 1. Generate IDs and identify candidate items
        candidate_ids: list[str] = [
            self.generate_doc_id(doc, idx) for idx, doc in enumerate(documents)
        ]

        # 2. Query existing documents in batches to avoid huge payloads
        existing_id_set: set[str] = set()
        chunk_size_check = 500
        for i in range(0, len(candidate_ids), chunk_size_check):
            sub_ids = candidate_ids[i : i + chunk_size_check]
            try:
                found = self.collection.get(ids=sub_ids)
                if found and "ids" in found:
                    existing_id_set.update(found["ids"])
            except Exception as e:
                logger.warning(f"Error checking existing IDs batch: {e}")

        # 3. Filter out duplicates
        docs_to_add: list[Document] = []
        ids_to_add: list[str] = []

        # Track IDs within this batch itself to prevent internal duplicates
        seen_in_batch: set[str] = set()

        for doc, doc_id in zip(documents, candidate_ids):
            if doc_id in existing_id_set or doc_id in seen_in_batch:
                continue
            seen_in_batch.add(doc_id)
            docs_to_add.append(doc)
            ids_to_add.append(doc_id)

        skipped_count = len(documents) - len(docs_to_add)
        if skipped_count > 0:
            logger.info(
                f"Skipped {skipped_count} duplicate documents already present in collection."
            )

        if not docs_to_add:
            logger.info("All documents already indexed; no new chunks to store.")
            return []

        logger.info(
            f"Ingesting {len(docs_to_add)} new document chunks into collection '{self.collection_name}'."
        )

        # 4. Compute embeddings
        texts = [doc.content for doc in docs_to_add]
        embeddings = self.embedding_manager.embed_batch(texts, batch_size=batch_size)

        # 5. Insert into ChromaDB in batches
        added_ids: list[str] = []
        total_to_add = len(docs_to_add)

        for i in range(0, total_to_add, batch_size):
            batch_slice = slice(i, i + batch_size)
            batch_docs = docs_to_add[batch_slice]
            batch_sub_ids = ids_to_add[batch_slice]
            batch_embs = [emb.tolist() for emb in embeddings[batch_slice]]
            batch_metas = [
                self._sanitize_metadata(doc.metadata) for doc in batch_docs
            ]
            batch_texts = [doc.content for doc in batch_docs]

            try:
                self.collection.add(
                    ids=batch_sub_ids,
                    embeddings=batch_embs,
                    metadatas=batch_metas,
                    documents=batch_texts,
                )
                added_ids.extend(batch_sub_ids)
            except Exception as insert_err:
                logger.error(f"Error inserting document batch {i}: {insert_err}")
                raise RuntimeError(
                    f"Failed inserting document batch into ChromaDB: {insert_err}"
                ) from insert_err

        logger.info(
            f"Successfully added {len(added_ids)} documents to collection '{self.collection_name}'. "
            f"Total collection count: {self.collection.count()}."
        )
        return added_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform semantic similarity search on the ChromaDB collection.

        Args:
            query: User search query or troubleshooting prompt.
            top_k: Number of most relevant documents to return.
            metadata_filter: Optional ChromaDB where-clause filter (e.g., {'source': 'manual.pdf'}).

        Returns:
            List of tuples (Document, similarity_score) sorted by descending similarity.
            Similarity scores range from 0.0 to 1.0 (higher is more relevant).
        """
        if not query.strip():
            return []

        total_docs = self.collection.count()
        if total_docs == 0:
            logger.warning(
                f"Collection '{self.collection_name}' is empty. Returning 0 search results."
            )
            return []

        # Generate query embedding
        query_emb = self.embedding_manager.embed_text(query).tolist()
        n_results = min(max(1, top_k), total_docs)

        try:
            raw_results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                where=metadata_filter if metadata_filter else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Error executing ChromaDB search: {e}")
            return []

        results: list[tuple[Document, float]] = []

        doc_lists = raw_results.get("documents", [[]])
        meta_lists = raw_results.get("metadatas", [[]])
        dist_lists = raw_results.get("distances", [[]])

        if not doc_lists or not doc_lists[0]:
            return []

        documents = doc_lists[0]
        metadatas = meta_lists[0] if meta_lists else [{}] * len(documents)
        distances = dist_lists[0] if dist_lists else [1.0] * len(documents)

        for text, meta, dist in zip(documents, metadatas, distances):
            # For cosine distance: distance = 1 - cosine_sim, where dist in [0, 2]
            # Convert to similarity score in [0, 1]
            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            doc = Document(content=text, metadata=dict(meta) if meta else {})
            results.append((doc, similarity_score))

        return results

    def get_all_documents(self) -> list[Document]:
        """Fetch all documents and metadatas currently stored in the collection.

        Enables initialization of secondary indices (such as BM25 for hybrid search).

        Returns:
            List of Document objects representing all indexed manual chunks.
        """
        count = self.collection.count()
        if count == 0:
            return []

        try:
            records = self.collection.get(include=["documents", "metadatas"])
            texts = records.get("documents", [])
            metadatas = records.get("metadatas", [])

            docs: list[Document] = []
            for text, meta in zip(texts, metadatas):
                docs.append(Document(content=text, metadata=dict(meta) if meta else {}))

            logger.debug(
                f"Retrieved {len(docs)} documents from ChromaDB collection '{self.collection_name}'."
            )
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve all documents from ChromaDB: {e}")
            return []

    def delete_collection(self) -> None:
        """Clear and delete the entire collection, resetting persistent state."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.warning(
                f"Collection '{self.collection_name}' has been cleared and recreated."
            )
        except Exception as e:
            logger.error(f"Error resetting collection '{self.collection_name}': {e}")
            raise

    def get_collection_stats(self) -> dict[str, Any]:
        """Retrieve operational statistics and configuration for the collection.

        Returns:
            Dictionary with document count, collection name, directory, and model details.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "num_documents": count,
            "persist_directory": str(self.persist_dir),
            "embedding_model": self.embedding_manager.model_name,
            "embedding_dimension": self.embedding_manager.dimension,
            "status": "active",
        }
