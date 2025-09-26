"""
RAG Engine for context generation and similarity merging.
"""

import logging
from dataclasses import dataclass
from typing import List
from openai import OpenAI

from error_log_monitor.config import SystemConfig
from error_log_monitor.opensearch_client import ErrorLog
from error_log_monitor.vector_db_client import VectorDBClient, VectorChunk
from error_log_monitor.embedding_service import EmbeddingService, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MergedIssue:
    """Merged similar error issue."""

    issue_id: str
    representative_log: ErrorLog
    similar_logs: List[ErrorLog]
    context: str
    occurrence_count: int
    time_span: str
    affected_services: List[str]


class RAGEngine:
    """RAG engine for context generation and similarity merging."""

    def __init__(self, config: SystemConfig, vector_db_client: VectorDBClient):
        """Initialize RAG engine."""
        self.config = config
        self.rag_config = config.rag
        self.vector_db = vector_db_client
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.embedding_service = EmbeddingService(model_name=config.vector_db.embedding_model)

    def generate_context(self, error_log: ErrorLog, similar_logs: List[ErrorLog]) -> str:
        """
        Generate context for error analysis using RAG.

        Args:
            error_log: Primary error log to analyze
            similar_logs: Similar error logs for context

        Returns:
            Generated context string
        """
        try:
            # Create context from similar logs
            context_parts = []

            # Add primary error details
            context_parts.append("Primary Error:")
            context_parts.append(f"- Message: {error_log.error_message}")
            context_parts.append(f"- Type: {error_log.error_type or 'Unknown'}")
            context_parts.append(f"- Service: {error_log.service or 'Unknown'}")
            context_parts.append(f"- Timestamp: {error_log.timestamp}")

            if error_log.traceback:
                context_parts.append(f"- Traceback: {error_log.traceback[:1000]}...")

            # Add similar errors context
            if similar_logs:
                context_parts.append(f"\nSimilar Errors ({len(similar_logs)} occurrences):")
                for i, log in enumerate(similar_logs[:5], 1):  # Limit to 5 similar logs
                    context_parts.append(f"{i}. {log.error_message[:100]}...")
                    context_parts.append(f"   Service: {log.service}, Time: {log.timestamp}")

            # Add service-specific context
            context_parts.append("\nVortexai Service Context:")
            context_parts.append("- Architecture: AWS Lambdas, ECS services, RDS, Milvus, OpenSearch")
            context_parts.append("- Critical APIs: UPDATE_CAMERAINFO, PARSE_METADATA")
            context_parts.append("- Database: PostgreSQL RDS with CAMERA_INFO and OBJECT_TRACE tables")

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return f"Context generation failed: {str(e)}"

    def merge_similar_issues(self, error_logs: List[ErrorLog]) -> List[MergedIssue]:
        """
        Merge similar error issues using RAG techniques with pre-calculated embeddings.

        Args:
            error_logs: List of error logs to merge

        Returns:
            List of merged issues
        """
        try:
            if not error_logs:
                return []

            logger.info(f"Pre-calculating embeddings for {len(error_logs)} error logs")

            # Pre-calculate all embeddings for efficient comparison
            embeddings_cache = self._calculate_embeddings_cache(error_logs)

            # Group similar logs using cached embeddings
            merged_issues = []
            processed_logs = set()

            for i, log in enumerate(error_logs):
                if log.message_id in processed_logs:
                    continue

                # Find similar logs using cached embeddings
                similar_logs = self._find_similar_logs_with_cache(
                    log, error_logs, processed_logs, embeddings_cache, max_similar_logs=len(error_logs)
                )

                # Create merged issue
                merged_issue = self._create_merged_issue(log, similar_logs)
                merged_issues.append(merged_issue)

                # Mark logs as processed
                processed_logs.add(log.message_id)
                for similar_log in similar_logs:
                    processed_logs.add(similar_log.message_id)

            logger.info(f"Merged {len(error_logs)} logs into {len(merged_issues)} issues")
            return merged_issues

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error merging similar issues due to embedding failure: {e}")
            raise  # Re-raise the exception instead of returning empty list
        except Exception as e:
            logger.error(f"Unexpected error merging similar issues: {e}")
            raise RuntimeError(f"Unexpected error in issue merging: {e}") from e

    def _calculate_error_log_embeddings(self, error_log: ErrorLog) -> List[float]:
        """Calculate embeddings for a single error log."""
        try:
            # Combine all text fields into one input
            text_input = (
                f"{error_log.error_message or ''} {error_log.error_type or ''} " f"{error_log.traceback or ''}"
            ).strip()

            # Check if text input is empty or only whitespace
            if not text_input or not text_input.strip():
                logger.warning(f"Empty text input for error log {error_log.message_id}, skipping embedding generation")
                raise ValueError("Cannot generate embedding for empty text input")

            # Generate single embedding for the combined text
            embedding = self.embedding_service.generate_embedding(text_input)
            return embedding
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error calculating error log embeddings: {e}", exc_info=True)
            raise  # Re-raise the exception instead of returning zero vector
        except Exception as e:
            logger.error(f"Unexpected error calculating error log embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error in embedding calculation: {e}") from e

    def _calculate_embeddings_cache(self, error_logs: List[ErrorLog]) -> dict:
        """
        Pre-calculate embeddings for all error logs to enable efficient similarity comparison.

        Args:
            error_logs: List of error logs to process

        Returns:
            Dictionary mapping log message_id to embeddings for error_message, error_type, and traceback

        Raises:
            ValueError: If any error log has empty text input
            RuntimeError: If embedding generation fails for any log
        """
        embeddings_cache = {}
        failed_logs = []

        for log in error_logs:
            try:
                embeddings_cache[log.message_id] = self._calculate_error_log_embeddings(log)
            except (ValueError, RuntimeError) as e:
                logger.error(f"Failed to calculate embedding for log {log.message_id}: {e}")
                failed_logs.append(log.message_id)
                # Continue processing other logs instead of failing completely
                continue

        if failed_logs:
            error_msg = f"Failed to calculate embeddings for {len(failed_logs)} logs: {failed_logs}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return embeddings_cache

    def _find_similar_logs_with_cache(
        self,
        primary_log: ErrorLog,
        all_logs: List[ErrorLog],
        processed_logs: set,
        embeddings_cache: dict,
        max_similar_logs: int = 5,
    ) -> List[ErrorLog]:
        """Find similar logs using cached embeddings for efficient comparison."""
        try:
            similar_logs = []
            primary_embeddings = embeddings_cache.get(primary_log.message_id)

            if not primary_embeddings:
                logger.warning(f"No cached embeddings found for log {primary_log.message_id}")
                return []

            for log in all_logs:
                if log.message_id != primary_log.message_id and log.message_id not in processed_logs:
                    # Check similarity using cached embeddings
                    are_similar = self._are_logs_similar_with_cache(primary_log, log, embeddings_cache)
                    if are_similar:
                        similar_logs.append(log)
                        processed_logs.add(log.message_id)

            return similar_logs[:max_similar_logs]

        except Exception as e:
            logger.error(f"Error finding similar logs with cache: {e}")
            return []

    def _are_logs_similar_with_cache(
        self,
        log1: ErrorLog,
        log2: ErrorLog,
        embeddings_cache: dict,
        threshold: float = 0.85,
    ) -> bool:
        """Check if two logs are similar using cached embeddings."""
        try:
            # Check if services match
            if log1.service and log2.service:
                if log1.service != log2.service:
                    logger.info(f"Services do not match: {log1.service} != {log2.service}")
                    return False

            # Get cached embeddings for log1
            log1_embeddings = embeddings_cache.get(log1.message_id)
            if not log1_embeddings:
                logger.warning(f"No cached embeddings found for log {log1.message_id}")
                return False

            # Get cached embeddings for log2
            log2_embeddings = embeddings_cache.get(log2.message_id)
            if not log2_embeddings:
                logger.warning(f"No cached embeddings found for log {log2.message_id}")
                return False

            # Compare embeddings similarity
            similarity = cosine_similarity(log1_embeddings, log2_embeddings)
            logger.info(f"  Similarity: {similarity:.3f} (threshold: {threshold})")
            return similarity >= threshold
        except Exception as e:
            logger.error(f"Error checking log similarity with cache: {e}", exc_info=True)
            return False

    def _create_chunks_from_logs(self, error_logs: List[ErrorLog]) -> List[VectorChunk]:
        """Create vector chunks from error logs."""
        chunks = []

        for log in error_logs:
            # Create chunk content
            content = f"Error: {log.error_message}"
            if log.traceback:
                content += f"\nTraceback: {log.traceback}"
            if log.error_type:
                content += f"\nType: {log.error_type}"

            # Create metadata
            metadata = {
                "message_id": log.message_id,
                "timestamp": log.timestamp.isoformat(),
                "service": log.service or "unknown",
                "error_type": log.error_type or "unknown",
                "site": log.site,
                "category": log.category or "unknown",
            }

            chunk = VectorChunk(content=content, metadata=metadata)
            chunks.append(chunk)

        return chunks

    def _find_similar_logs(
        self,
        primary_log: ErrorLog,
        all_logs: List[ErrorLog],
        processed_logs: set,
        max_similar_logs: int = 5,
    ) -> List[ErrorLog]:
        """Find similar logs using vector similarity."""
        try:
            similar_logs = []
            for log in all_logs:
                if log.message_id != primary_log.message_id and log.message_id not in processed_logs:
                    # Apply similarity filtering based on error type and message patterns
                    are_similar = self._are_logs_similar(primary_log, log)
                    if are_similar:
                        similar_logs.append(log)
                        processed_logs.add(log.message_id)

            return similar_logs[:max_similar_logs]

        except Exception as e:
            logger.error(f"Error finding similar logs: {e}")
            return []

    def _are_logs_similar(self, log1: ErrorLog, log2: ErrorLog) -> bool:
        """Check if two logs are similar based on embedding similarity."""
        try:
            # Check if error types match
            # if log1.error_type and log2.error_type:
            #     if log1.error_type != log2.error_type:
            #         logger.info(f"Error types do not match: {log1.error_type} != {log2.error_type}")
            #         return False

            # Check if services match
            if log1.service and log2.service:
                if log1.service != log2.service:
                    logger.info(f"Services do not match: {log1.service} != {log2.service}")
                    return False

            # Use embedding similarity for error messages
            return self._are_messages_similar_by_embedding(log1, log2)

        except Exception as e:
            logger.error(f"Error checking log similarity: {e}")
            return False

    def _are_messages_similar_by_embedding(self, log1: ErrorLog, log2: ErrorLog) -> bool:
        """Check if two error messages are similar using embedding similarity."""
        try:
            # Generate embeddings for both messages
            embedding1 = self.embedding_service.generate_embedding(log1.error_message)
            embedding2 = self.embedding_service.generate_embedding(log2.error_message)
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)
            if similarity >= 0.99:
                logger.info(f"  Similarity: {similarity:.3f}")
                if log1.traceback and log2.traceback:
                    traceback_embedding1 = self.embedding_service.generate_embedding(log1.traceback[:1000])
                    traceback_embedding2 = self.embedding_service.generate_embedding(log2.traceback[:1000])
                    traceback_similarity = cosine_similarity(traceback_embedding1, traceback_embedding2)
                    logger.info(f"  Traceback Similarity: {traceback_similarity:.3f}")

                    return traceback_similarity >= 0.9
                return True

            embedding3 = self.embedding_service.generate_embedding(log1.error_message[:100])
            embedding4 = self.embedding_service.generate_embedding(log2.error_message[:100])
            similarity = cosine_similarity(embedding3, embedding4)
            if similarity >= 0.7:
                logger.info(f"2nd  Similarity: {similarity:.3f}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}", exc_info=True)
            return False

    def _create_merged_issue(self, representative_log: ErrorLog, similar_logs: List[ErrorLog]) -> MergedIssue:
        """Create merged issue from representative log and similar logs."""
        try:
            # Generate context
            context = self.generate_context(representative_log, similar_logs)

            # Calculate time span
            all_logs = [representative_log] + similar_logs
            timestamps = [log.timestamp for log in all_logs]
            time_span = f"{min(timestamps)} to {max(timestamps)}"

            # Get affected services
            affected_services = list(set(log.service for log in all_logs if log.service))

            # Create issue ID
            issue_id = f"issue_{hash(representative_log.error_message) % 10000}"

            return MergedIssue(
                issue_id=issue_id,
                representative_log=representative_log,
                similar_logs=similar_logs,
                context=context,
                occurrence_count=len(all_logs),
                time_span=time_span,
                affected_services=affected_services,
            )

        except Exception as e:
            logger.error(f"Error creating merged issue: {e}")
            # Return minimal merged issue
            return MergedIssue(
                issue_id=f"issue_{hash(representative_log.error_message) % 10000}",
                representative_log=representative_log,
                similar_logs=similar_logs,
                context="Context generation failed",
                occurrence_count=1,
                time_span=str(representative_log.timestamp),
                affected_services=[representative_log.service or "unknown"],
            )

    def retrieve_context_for_analysis(self, error_log: ErrorLog) -> str:
        """
        Retrieve relevant context for error analysis.

        Args:
            error_log: Error log to analyze

        Returns:
            Retrieved context string
        """
        try:
            # Create query from error log
            query = f"Error: {error_log.error_message}"
            if error_log.traceback:
                query += f" Traceback: {error_log.traceback}"

            # Search for similar chunks
            similar_chunks = self.vector_db.search_similar_chunks(
                query=query, n_results=self.rag_config.max_retrieved_chunks, where={"site": error_log.site}
            )

            # Build context from retrieved chunks
            context_parts = []
            context_parts.append("Retrieved Context from Similar Errors:")

            for i, chunk in enumerate(similar_chunks, 1):
                context_parts.append(f"{i}. {chunk.content[:200]}...")
                context_parts.append(f"   Service: {chunk.metadata.get('service', 'unknown')}")
                context_parts.append(f"   Time: {chunk.metadata.get('timestamp', 'unknown')}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Context retrieval failed"
