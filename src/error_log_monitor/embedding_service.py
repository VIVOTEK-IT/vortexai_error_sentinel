"""
Embedding service for generating and comparing text embeddings.

This module provides a centralized service for handling all embedding-related operations,
including text embedding generation, similarity calculation, and batch processing.
"""

import logging
import os
from typing import List, Optional, Tuple
import numpy as np

# Try to import OpenAI
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)


class ExceptionEmbeddingContextExceeded(RuntimeError):
    """Exception raised when embedding context exceeds the maximum allowed length."""

    pass


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize vector to unit vector (length = 1).

    Args:
        vector: Raw vector to normalize

    Returns:
        Normalized unit vector

    Raises:
        ValueError: If vector is zero vector or invalid
        RuntimeError: If normalization fails
    """
    try:
        # Convert to numpy array for efficient calculation
        vec = np.array(vector, dtype=np.float32)

        # Calculate L2 norm
        norm = np.linalg.norm(vec)

        # Check for zero vector
        if norm == 0:
            error_msg = "Cannot normalize zero vector to unit vector"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize to unit vector
        normalized = vec / norm

        return normalized.tolist()

    except ValueError:
        # Re-raise ValueError for zero vectors
        raise
    except Exception as e:
        error_msg = f"Error normalizing vector: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def validate_unit_vector(vector: List[float], tolerance: float = 1e-6) -> bool:
    """
    Validate that vector is a unit vector.

    Args:
        vector: Vector to validate
        tolerance: Tolerance for magnitude check

    Returns:
        True if vector is unit vector, False otherwise
    """
    try:
        vec = np.array(vector, dtype=np.float32)
        magnitude = np.linalg.norm(vec)
        return abs(magnitude - 1.0) <= tolerance

    except Exception as e:
        logger.error(f"Error validating unit vector: {e}")
        return False


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between 0 and 1.
    """
    try:
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)

    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


class EmbeddingService:
    """Service for handling text embeddings and similarity calculations."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model_name: OpenAI embedding model name.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required but not available. Please install openai>=1.3.0")

        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"Initialized EmbeddingService with model: {self.model_name}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of normalized unit vector embeddings (each vector is a list of floats).

        Raises:
            ValueError: If texts list is empty or contains empty strings
            RuntimeError: If OpenAI client is not initialized or embedding generation fails
        """
        if not self.client:
            # Throw exception if OpenAI is not available
            error_msg = "OpenAI client not initialized, cannot generate embeddings"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

        if not texts or len(texts) == 0 or texts == [""]:
            # Throw exception if no texts provided
            error_msg = "No texts provided for embedding generation"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

        try:
            response = self.client.embeddings.create(model=self.model_name, input=texts)
            raw_embeddings = [embedding.embedding for embedding in response.data]

            # Normalize all embeddings to unit vectors
            normalized_embeddings = [normalize_vector(embedding) for embedding in raw_embeddings]

            logger.debug(f"Generated {len(normalized_embeddings)} normalized embeddings")
            return normalized_embeddings
        except Exception as e:
            if hasattr(e, 'message'):
                if "This model's maximum context length is" in e.message:
                    logger.error(e.message, exc_info=True)
                    raise ExceptionEmbeddingContextExceeded(e.message) from e
            elif "This model's maximum context length is" in str(e):
                logger.error(str(e), exc_info=True)
                raise ExceptionEmbeddingContextExceeded(e.message) from e
            else:
                error_msg = f"Failed to generate embeddings: {e}. Input: {texts}"
                logger.error(error_msg, exc_info=True)
                # Re-raise the exception instead of returning dummy embeddings
                raise RuntimeError(error_msg) from e

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Normalized unit vector embedding as a list of floats.

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If OpenAI client is not initialized or embedding generation fails
        """
        if not text or not text.strip():
            error_msg = "Empty or invalid text provided for embedding generation"
            logger.error(error_msg)
            raise ValueError(error_msg)

        embeddings = self.generate_embeddings([text])
        return embeddings[0]

    def are_texts_similar(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.7,
    ) -> Tuple[bool, float]:
        """
        Check if two texts are similar using embedding similarity.

        Args:
            text1: First text to compare.
            text2: Second text to compare.
            threshold: Similarity threshold for main text comparison.
            use_traceback: Whether to also check traceback similarity.
            traceback_threshold: Similarity threshold for traceback comparison.

        Returns:
            Tuple of (is_similar, similarity_score).
        """
        try:
            # Generate embeddings for both texts
            embedding1 = self.generate_embedding(text1)
            embedding2 = self.generate_embedding(text2)

            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)

            if similarity >= threshold:
                return True, similarity

            # Check with truncated text for partial matches
            if len(text1) > 1000 and len(text2) > 1000:
                embedding3 = self.generate_embedding(text1[:1000])
                embedding4 = self.generate_embedding(text2[:1000])
                similarity = cosine_similarity(embedding3, embedding4)

                if similarity >= threshold * 0.8:
                    logger.info(f"Truncated similarity: {similarity:.3f}")
                    return True, similarity

            return similarity >= threshold, similarity

        except Exception as e:
            logger.error(f"Error checking text similarity: {e}", exc_info=True)
            return False, 0.0

    def batch_similarity_check(self, texts: List[str], threshold: float = 0.7) -> List[Tuple[int, int, float]]:
        """
        Check similarity between all pairs of texts in a batch.

        Args:
            texts: List of texts to compare.
            threshold: Similarity threshold.

        Returns:
            List of tuples (index1, index2, similarity) for similar pairs.
        """
        if len(texts) < 2:
            return []

        # Generate all embeddings at once
        embeddings = self.generate_embeddings(texts)
        similar_pairs = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))

        return similar_pairs

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension (1536 for text-embedding-3-small).
        """
        return 1536  # Standard dimension for text-embedding-3-small

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an existing embedding to unit vector.

        Args:
            embedding: Raw embedding vector to normalize

        Returns:
            Normalized unit vector

        Raises:
            ValueError: If embedding is zero vector or invalid
            RuntimeError: If normalization fails
        """
        return normalize_vector(embedding)

    def validate_unit_vector(self, embedding: List[float], tolerance: float = 1e-6) -> bool:
        """
        Validate that an embedding is a unit vector.

        Args:
            embedding: Vector to validate
            tolerance: Tolerance for magnitude check

        Returns:
            True if vector is unit vector, False otherwise
        """
        return validate_unit_vector(embedding, tolerance)

    def test_connection(self) -> bool:
        """
        Test the connection to OpenAI API.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            # Try to generate a simple embedding
            test_embedding = self.generate_embedding("test")
            return len(test_embedding) == self.get_embedding_dimension()
        except Exception as e:
            logger.error(f"Embedding service connection test failed: {e}", exc_info=True)
            return False


def create_embedding_service(
    api_key: Optional[str] = None, model_name: str = "text-embedding-3-small"
) -> EmbeddingService:
    """
    Factory function to create an embedding service.

    Args:
        api_key: OpenAI API key.
        model_name: OpenAI model name.

    Returns:
        EmbeddingService instance.
    """

    return EmbeddingService(api_key=api_key, model_name=model_name)
