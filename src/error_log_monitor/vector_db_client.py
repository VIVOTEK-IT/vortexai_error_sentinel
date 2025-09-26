"""
Vector database client for RAG context storage and retrieval.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from error_log_monitor.embedding_service import EmbeddingService

# Disable ChromaDB telemetry to avoid posthog errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from error_log_monitor.config import VectorDBConfig

# Suppress all ChromaDB telemetry errors
chromadb_logger = logging.getLogger('chromadb.telemetry.posthog')
chromadb_logger.setLevel(logging.CRITICAL)

# Suppress product telemetry as well
chromadb_product_logger = logging.getLogger('chromadb.telemetry.product.posthog')
chromadb_product_logger.setLevel(logging.CRITICAL)

# Suppress all telemetry loggers
for logger_name in [
    'chromadb.telemetry',
    'chromadb.telemetry.posthog',
    'chromadb.telemetry.product',
    'chromadb.telemetry.product.posthog',
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


@dataclass
class VectorChunk:
    """Vector chunk for storage in vector database."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorDBClient:
    """Vector database client for RAG operations."""

    def __init__(self, config: VectorDBConfig):
        """Initialize vector database client."""
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_service = EmbeddingService(model_name=config.embedding_model)
        self.embedding_function = None
        self._connect()

    def _connect(self):
        """Connect to vector database."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.config.persist_directory, exist_ok=True)

            # Initialize ChromaDB client with telemetry disabled
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory, settings=chromadb.Settings(anonymized_telemetry=False)
            )

            # Initialize embedding function using EmbeddingService
            try:
                self.embedding_function = self.embedding_service.generate_embeddings
                logger.info(f"Initialized embedding function with model: {self.config.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
                raise

            # Get or create collection with embedding function
            try:
                # Try to get existing collection with embedding function
                self.collection = self.client.get_collection(
                    name=self.config.collection_name, embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {self.config.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it with embedding function
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.config.distance_metric},
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
            except Exception as e:
                logger.warning(f"Error accessing collection: {e}")
                # Force create a new collection by deleting the old one
                try:
                    # Delete existing collection if it exists
                    self.client.delete_collection(name=self.config.collection_name)
                    logger.info(f"Deleted existing collection: {self.config.collection_name}")
                except Exception:
                    pass
                # Create new collection with embedding function
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.config.distance_metric},
                )
                logger.info(f"Created new collection after cleanup: {self.config.collection_name}")

            logger.info(f"Connected to vector database: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}", exc_info=True)
            raise

    def store_chunks(self, chunks: List[VectorChunk]) -> bool:
        """
        Store vector chunks in the database.

        Args:
            chunks: List of vector chunks to store

        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            logger.error("Not connected to vector database", exc_info=True)
            return False

        try:
            # Prepare data for storage
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Convert metadata values to strings for ChromaDB compatibility
                metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, list):
                        metadata[key] = ",".join(str(v) for v in value)
                    else:
                        metadata[key] = str(value)

                documents.append(chunk.content)
                metadatas.append(metadata)
                ids.append(f"chunk_{i}_{hash(chunk.content) % 10000}")

            # Store in collection
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

            logger.info(f"Stored {len(chunks)} chunks in vector database")
            return True

        except Exception as e:
            logger.error(f"Error storing chunks: {e}", exc_info=True)
            return False

    def search_similar_chunks(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[VectorChunk]:
        """
        Search for similar chunks.

        Args:
            query: Query string
            n_results: Number of results to return
            where: Metadata filter

        Returns:
            List of similar vector chunks
        """
        if not self.collection:
            logger.error("Not connected to vector database", exc_info=True)
            return []

        try:
            # Perform similarity search using embedding function
            results = self.collection.query(
                query_texts=[query], n_results=n_results, where=where, include=['documents', 'metadatas', 'distances']
            )

            # Convert results to VectorChunk objects
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
                ):
                    chunk = VectorChunk(content=doc, metadata=metadata, embedding=None)  # Not needed for retrieval
                    chunks.append(chunk)

            logger.info(f"Found {len(chunks)} similar chunks for query")
            return chunks

        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}", exc_info=True)
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            return {"error": "Not connected to vector database"}

        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.config.collection_name,
                "distance_metric": self.config.distance_metric,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}", exc_info=True)
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        if not self.collection:
            logger.error("Not connected to vector database", exc_info=True)
            return False

        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])

            logger.info("Cleared vector database collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}", exc_info=True)
            return False

    def test_connection(self) -> bool:
        """Test vector database connection."""
        try:
            if not self.collection:
                return False
            self.get_collection_stats()
            return True
        except Exception as e:
            logger.error(f"Vector database connection test failed: {e}", exc_info=True)
            return False
