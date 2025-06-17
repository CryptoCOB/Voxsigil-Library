"""
Base Interface Definitions
=========================

Core interface contracts that define fundamental capabilities across
the VoxSigil Library. These interfaces serve as the foundation for
all specialized implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable


class BaseRagInterface(ABC):
    """
    Unified RAG (Retrieval-Augmented Generation) Interface
    
    Consolidates all RAG-related functionality across modules:
    - Document retrieval and indexing
    - Context management and ranking
    - Query processing and augmentation
    """
    
    @abstractmethod
    async def retrieve_documents(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        pass
    
    @abstractmethod
    async def index_document(
        self, 
        document: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Index a document for retrieval."""
        pass
    
    @abstractmethod
    async def augment_query(
        self, 
        query: str, 
        context: List[Dict[str, Any]]
    ) -> str:
        """Augment query with retrieved context."""
        pass
    
    @abstractmethod
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""
        pass


class BaseLlmInterface(ABC):
    """
    Unified LLM (Large Language Model) Interface
    
    Standardizes interaction with various LLM providers and models:
    - Text generation and completion
    - Model configuration and parameters
    - Token management and optimization
    """
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text completion for a prompt."""
        pass
    
    @abstractmethod
    async def generate_streaming(
        self,
        prompt: str,
        callback: Callable[[str], None],
        **kwargs
    ) -> None:
        """Generate streaming text with callback."""
        pass
    
    @abstractmethod
    async def get_embeddings(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Get embeddings for input texts."""
        pass
    
    @abstractmethod
    async def validate_model(self) -> bool:
        """Validate model availability and configuration."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass


class BaseMemoryInterface(ABC):
    """
    Unified Memory Management Interface
    
    Handles persistent and temporary memory across the system:
    - Conversation history and context
    - Long-term knowledge storage
    - Memory retrieval and cleanup
    """
    
    @abstractmethod
    async def store_memory(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Store a memory item."""
        pass
    
    @abstractmethod
    async def retrieve_memory(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Retrieve a memory item."""
        pass
    
    @abstractmethod
    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search memories by content or metadata."""
        pass
    
    @abstractmethod
    async def delete_memory(self, key: str) -> bool:
        """Delete a specific memory."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired memories, return count deleted."""
        pass
    
    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        pass


class BaseAgentInterface(ABC):
    """
    Unified Agent Interface
    
    Defines core agent capabilities and lifecycle:
    - Task execution and planning
    - Tool usage and coordination
    - State management and persistence
    """
    
    @abstractmethod
    async def execute_task(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a specific task."""
        pass
    
    @abstractmethod
    async def plan_execution(
        self,
        goal: str,
        constraints: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Plan task execution steps."""
        pass
    
    @abstractmethod
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        pass
    
    @abstractmethod
    async def save_state(self) -> Dict[str, Any]:
        """Save current agent state."""
        pass
    
    @abstractmethod
    async def restore_state(self, state: Dict[str, Any]) -> bool:
        """Restore agent from saved state."""
        pass


class BaseModelInterface(ABC):
    """
    Unified Model Management Interface
    
    Handles model lifecycle and configuration:
    - Model loading and initialization
    - Performance monitoring
    - Resource management
    """
    
    @abstractmethod
    async def load_model(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Load a model from path."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> bool:
        """Unload current model."""
        pass
    
    @abstractmethod
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and metrics."""
        pass
    
    @abstractmethod
    async def optimize_model(
        self,
        optimization_config: Dict[str, Any]
    ) -> bool:
        """Apply model optimizations."""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass
