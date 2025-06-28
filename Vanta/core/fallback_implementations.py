"""
Centralized Fallback Implementations
===================================

This module consolidates all mock, stub, and fallback implementations
that were previously scattered across the codebase. Provides reliable
fallback behavior when primary services are unavailable.

Architecture:
- Unified fallback registry
- Graceful degradation strategies
- Performance monitoring for fallbacks
"""

# HOLO-1.5 Registration System
try:
    from Voxsigil_Library.core.vanta_registration import vanta_core_module
except ImportError:
    # Fallback registration decorator for development
    def vanta_core_module(cls):
        return cls


import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ..interfaces import (
    BaseLlmInterface,
    BaseMemoryInterface,
    BaseRagInterface,
    FallbackProtocol,
)


@vanta_core_module
class FallbackRegistry:
    """
    Central registry for all fallback implementations.
    Manages selection and routing to appropriate fallbacks.
    """

    def __init__(self):
        self._fallbacks: Dict[str, List[FallbackProtocol]] = {}
        self._usage_stats: Dict[str, Dict[str, Any]] = {}
        self._logger = logging.getLogger(__name__)

    def register_fallback(self, service_type: str, fallback: FallbackProtocol) -> None:
        """Register a fallback implementation for a service type."""
        if service_type not in self._fallbacks:
            self._fallbacks[service_type] = []

        self._fallbacks[service_type].append(fallback)
        self._fallbacks[service_type].sort(
            key=lambda f: f.reliability_score, reverse=True
        )

        self._logger.info(
            f"Registered fallback for {service_type}: {fallback.fallback_type}"
        )

    async def get_fallback(
        self, service_type: str, request_type: str, payload: Dict[str, Any]
    ) -> Optional[FallbackProtocol]:
        """Get best available fallback for service type and request."""
        if service_type not in self._fallbacks:
            return None

        for fallback in self._fallbacks[service_type]:
            if await fallback.can_handle_request(request_type, payload):
                # Update usage stats
                fallback_id = f"{service_type}:{fallback.fallback_type}"
                if fallback_id not in self._usage_stats:
                    self._usage_stats[fallback_id] = {
                        "usage_count": 0,
                        "last_used": None,
                        "success_rate": 1.0,
                    }

                self._usage_stats[fallback_id]["usage_count"] += 1
                self._usage_stats[fallback_id]["last_used"] = datetime.utcnow()

                return fallback

        return None

    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get fallback usage statistics."""
        return self._usage_stats.copy()


class FallbackRagInterface(BaseRagInterface):
    """
    Fallback RAG implementation providing basic functionality
    when primary RAG services are unavailable.
    """

    def __init__(self):
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "documents_indexed": 0,
            "queries_processed": 0,
            "retrieval_calls": 0,
        }
        self.reliability_score = 0.7
        self.fallback_type = "basic_rag"

    async def retrieve_documents(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Basic text matching for document retrieval."""
        self._stats["retrieval_calls"] += 1

        query_lower = query.lower()
        results = []

        for doc_id, doc in self._documents.items():
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()

            # Simple keyword matching
            score = 0
            for word in query_lower.split():
                if word in content:
                    score += content.count(word) * 0.1
                if word in title:
                    score += title.count(word) * 0.3

            if score > 0:
                results.append(
                    {
                        "document_id": doc_id,
                        "content": doc["content"],
                        "score": score,
                        "metadata": doc.get("metadata", {}),
                    }
                )

        # Sort by score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    async def index_document(
        self, document: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Simple document storage for fallback indexing."""
        doc_id = f"fallback_doc_{len(self._documents)}"

        self._documents[doc_id] = {
            "content": document.get("content", ""),
            "title": document.get("title", ""),
            "metadata": metadata or {},
            "indexed_at": datetime.utcnow().isoformat(),
        }

        self._stats["documents_indexed"] += 1
        return doc_id

    async def augment_query(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Basic query augmentation with context."""
        if not context:
            return query

        context_text = "\n".join(
            [doc.get("content", "")[:200] + "..." for doc in context[:3]]
        )

        return f"Context: {context_text}\n\nQuery: {query}"

    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get fallback RAG statistics."""
        return {
            **self._stats,
            "total_documents": len(self._documents),
            "fallback_type": self.fallback_type,
            "reliability_score": self.reliability_score,
        }

    async def can_handle_request(
        self, request_type: str, payload: Dict[str, Any]
    ) -> bool:
        """Check if this fallback can handle the request."""
        supported_requests = [
            "retrieve_documents",
            "index_document",
            "augment_query",
            "get_retrieval_stats",
        ]
        return request_type in supported_requests

    async def execute_fallback(
        self, request_type: str, payload: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fallback RAG operation."""
        try:
            if request_type == "retrieve_documents":
                result = await self.retrieve_documents(**payload)
            elif request_type == "index_document":
                result = await self.index_document(**payload)
            elif request_type == "augment_query":
                result = await self.augment_query(**payload)
            elif request_type == "get_retrieval_stats":
                result = await self.get_retrieval_stats()
            else:
                raise ValueError(f"Unsupported request type: {request_type}")

            return {
                "success": True,
                "result": result,
                "fallback_used": True,
                "fallback_type": self.fallback_type,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True,
                "fallback_type": self.fallback_type,
            }

    async def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback usage and performance metrics."""
        return {
            "fallback_type": self.fallback_type,
            "reliability_score": self.reliability_score,
            "usage_stats": self._stats,
            "total_documents": len(self._documents),
        }


class FallbackLlmInterface(BaseLlmInterface):
    """
    Fallback LLM implementation providing basic text generation
    when primary LLM services are unavailable.
    """

    def __init__(self):
        self._generation_stats = {
            "requests_processed": 0,
            "tokens_generated": 0,
            "errors": 0,
        }
        self.reliability_score = 0.5
        self.fallback_type = "template_llm"

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate template-based responses."""
        self._generation_stats["requests_processed"] += 1

        # Simple template responses based on prompt patterns
        prompt_lower = prompt.lower()

        if "question" in prompt_lower or "?" in prompt:
            response = "I understand you're asking a question. While I'm operating in fallback mode with limited capabilities, I can try to provide a basic response based on the information available."
        elif "code" in prompt_lower or "function" in prompt_lower:
            response = "I notice you're asking about code. In fallback mode, I can provide basic programming guidance, but for detailed code generation, please ensure the primary LLM service is available."
        elif "explain" in prompt_lower or "describe" in prompt_lower:
            response = "I can provide a basic explanation. However, for more detailed and accurate explanations, the primary LLM service would be more suitable."
        else:
            response = "I'm currently operating in fallback mode with limited capabilities. I can process your request, but the response quality may be reduced compared to the primary service."

        # Apply basic token limiting
        if max_tokens:
            words = response.split()
            # Rough token approximation (1 token ≈ 0.75 words)
            max_words = int(max_tokens * 0.75)
            if len(words) > max_words:
                response = " ".join(words[:max_words]) + "..."

        estimated_tokens = len(response.split()) * 1.3  # Rough token estimate
        self._generation_stats["tokens_generated"] += int(estimated_tokens)

        return response

    async def generate_streaming(
        self, prompt: str, callback: Callable[[str], None], **kwargs
    ) -> None:
        """Simulate streaming by sending response in chunks."""
        response = await self.generate_text(prompt, **kwargs)

        # Split response into chunks and send with delay
        words = response.split()
        chunk_size = 5  # Words per chunk

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            callback(chunk + " ")
            await asyncio.sleep(0.1)  # Simulate streaming delay

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple hash-based embeddings."""
        embeddings = []

        for text in texts:
            # Simple character-based embedding (not semantic)
            embedding = []
            for i in range(384):  # Standard embedding dimension
                char_sum = sum(ord(c) for c in text) if text else 0
                value = ((char_sum + i) % 256) / 255.0 - 0.5
                embedding.append(value)

            embeddings.append(embedding)

        return embeddings

    async def validate_model(self) -> bool:
        """Fallback model is always available."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get fallback model information."""
        return {
            "model_name": "fallback_template_model",
            "model_type": "template_based",
            "capabilities": ["text_generation", "basic_embeddings"],
            "limitations": [
                "Template-based responses only",
                "No semantic understanding",
                "Limited context awareness",
            ],
            "reliability_score": self.reliability_score,
            "fallback_mode": True,
        }

    async def can_handle_request(
        self, request_type: str, payload: Dict[str, Any]
    ) -> bool:
        """Check if this fallback can handle the request."""
        supported_requests = [
            "generate_text",
            "generate_streaming",
            "get_embeddings",
            "validate_model",
            "get_model_info",
        ]
        return request_type in supported_requests

    async def execute_fallback(
        self, request_type: str, payload: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fallback LLM operation."""
        try:
            if request_type == "generate_text":
                result = await self.generate_text(**payload)
            elif request_type == "get_embeddings":
                result = await self.get_embeddings(**payload)
            elif request_type == "validate_model":
                result = await self.validate_model()
            elif request_type == "get_model_info":
                result = self.get_model_info()
            else:
                raise ValueError(f"Unsupported request type: {request_type}")

            return {
                "success": True,
                "result": result,
                "fallback_used": True,
                "fallback_type": self.fallback_type,
            }

        except Exception as e:
            self._generation_stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True,
                "fallback_type": self.fallback_type,
            }

    async def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback usage and performance metrics."""
        return {
            "fallback_type": self.fallback_type,
            "reliability_score": self.reliability_score,
            "generation_stats": self._generation_stats,
        }


class FallbackMemoryInterface(BaseMemoryInterface):
    """
    Fallback memory implementation using in-memory storage
    when primary memory services are unavailable.
    """

    def __init__(self):
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "items_stored": 0,
            "items_retrieved": 0,
            "items_deleted": 0,
            "searches_performed": 0,
        }
        self.reliability_score = 0.6
        self.fallback_type = "in_memory"

    async def store_memory(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store memory item in fallback storage."""
        try:
            expiry_time = None
            if ttl:
                expiry_time = (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()

            self._memory_store[key] = {
                "value": value,
                "metadata": metadata or {},
                "stored_at": datetime.utcnow().isoformat(),
                "expires_at": expiry_time,
                "access_count": 0,
            }

            self._stats["items_stored"] += 1
            return True

        except Exception:
            return False

    async def retrieve_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve memory item from fallback storage."""
        self._stats["items_retrieved"] += 1

        if key not in self._memory_store:
            return default

        item = self._memory_store[key]

        # Check expiry
        if item.get("expires_at"):
            expiry = datetime.fromisoformat(item["expires_at"])
            if datetime.utcnow() > expiry:
                del self._memory_store[key]
                return default

        # Update access count
        item["access_count"] += 1
        item["last_accessed"] = datetime.utcnow().isoformat()

        return item["value"]

    async def search_memories(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search memories using simple text matching."""
        self._stats["searches_performed"] += 1

        results = []
        query_lower = query.lower()

        for key, item in self._memory_store.items():
            # Check expiry
            if item.get("expires_at"):
                expiry = datetime.fromisoformat(item["expires_at"])
                if datetime.utcnow() > expiry:
                    continue

            # Simple text search in value and metadata
            value_str = str(item["value"]).lower()
            metadata_str = str(item.get("metadata", {})).lower()

            score = 0
            for word in query_lower.split():
                if word in value_str:
                    score += value_str.count(word)
                if word in metadata_str:
                    score += metadata_str.count(word) * 0.5
                if word in key.lower():
                    score += key.lower().count(word) * 2

            if score > 0:
                results.append(
                    {
                        "key": key,
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "score": score,
                        "stored_at": item["stored_at"],
                    }
                )

        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for filter_key, filter_value in filters.items():
                    if filter_key not in result["metadata"]:
                        match = False
                        break
                    if result["metadata"][filter_key] != filter_value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            results = filtered_results

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def delete_memory(self, key: str) -> bool:
        """Delete memory item from fallback storage."""
        if key in self._memory_store:
            del self._memory_store[key]
            self._stats["items_deleted"] += 1
            return True
        return False

    async def cleanup_expired(self) -> int:
        """Clean up expired memories."""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, item in self._memory_store.items():
            if item.get("expires_at"):
                expiry = datetime.fromisoformat(item["expires_at"])
                if current_time > expiry:
                    expired_keys.append(key)

        for key in expired_keys:
            del self._memory_store[key]

        return len(expired_keys)

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get fallback memory statistics."""
        return {
            **self._stats,
            "total_items": len(self._memory_store),
            "fallback_type": self.fallback_type,
            "reliability_score": self.reliability_score,
        }

    async def can_handle_request(
        self, request_type: str, payload: Dict[str, Any]
    ) -> bool:
        """Check if this fallback can handle the request."""
        supported_requests = [
            "store_memory",
            "retrieve_memory",
            "search_memories",
            "delete_memory",
            "cleanup_expired",
            "get_memory_stats",
        ]
        return request_type in supported_requests

    async def execute_fallback(
        self, request_type: str, payload: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fallback memory operation."""
        try:
            if request_type == "store_memory":
                result = await self.store_memory(**payload)
            elif request_type == "retrieve_memory":
                result = await self.retrieve_memory(**payload)
            elif request_type == "search_memories":
                result = await self.search_memories(**payload)
            elif request_type == "delete_memory":
                result = await self.delete_memory(**payload)
            elif request_type == "cleanup_expired":
                result = await self.cleanup_expired()
            elif request_type == "get_memory_stats":
                result = await self.get_memory_stats()
            else:
                raise ValueError(f"Unsupported request type: {request_type}")

            return {
                "success": True,
                "result": result,
                "fallback_used": True,
                "fallback_type": self.fallback_type,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True,
                "fallback_type": self.fallback_type,
            }

    async def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback usage and performance metrics."""
        return {
            "fallback_type": self.fallback_type,
            "reliability_score": self.reliability_score,
            "memory_stats": self._stats,
            "total_items": len(self._memory_store),
        }


# Global fallback registry instance
fallback_registry = FallbackRegistry()


# Register default fallback implementations
async def initialize_fallbacks():
    """Initialize and register default fallback implementations."""
    fallback_registry.register_fallback("rag", FallbackRagInterface())
    fallback_registry.register_fallback("llm", FallbackLlmInterface())
    fallback_registry.register_fallback("memory", FallbackMemoryInterface())
