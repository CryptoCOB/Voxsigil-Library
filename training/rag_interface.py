"""
VoxSigil Training RAG Interface

Complete implementation of the RAG interface for VoxSigil training system integration.
Now unified with Vanta interface system for better modularity.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger_rag_interface = logging.getLogger("VoxSigilSupervisor.RAG")

# Import unified interfaces from Vanta
try:
    from Vanta.interfaces import BaseRagInterface
    VANTA_INTERFACES_AVAILABLE = True
    logger_rag_interface.info("Using unified Vanta RAG interfaces")
except ImportError:
    # Import unified interface from Vanta
    from Vanta.interfaces.base_interfaces import BaseRagInterface
    VANTA_INTERFACES_AVAILABLE = True
    logger_rag_interface.info("Successfully imported BaseRagInterface from Vanta")

# Try to import VoxSigilRAG
VOXSIGIL_RAG_AVAILABLE = False
try:
    # Try multiple import paths
    try:
        from BLT.voxsigil_rag import VoxSigilRAG as ImportedVoxSigilRAG

        VOXSIGIL_RAG_AVAILABLE = True
        logger_rag_interface.info(
            "Successfully imported VoxSigilRAG from Voxsigil_Library"
        )
    except ImportError:
        try:
            from BLT.voxsigil_rag import VoxSigilRAG as ImportedVoxSigilRAG

            VOXSIGIL_RAG_AVAILABLE = True
            logger_rag_interface.info("Successfully imported VoxSigilRAG directly")
        except ImportError:
            VOXSIGIL_RAG_AVAILABLE = False
except Exception:
    VOXSIGIL_RAG_AVAILABLE = False

if not VOXSIGIL_RAG_AVAILABLE:
    # Create a dummy VoxSigilRAG if not found
    class VoxSigilRAG:
        def __init__(self, *args, **kwargs):
            logger_rag_interface.warning(
                "Using dummy VoxSigilRAG - RAG functionality will be limited"
            )

        def inject_voxsigil_context(self, *args, **kwargs):
            return "", []

        def create_rag_context(self, *args, **kwargs):
            return "", []

    logger_rag_interface.warning(
        "VoxSigilRAG not available, using dummy implementation"
    )


class SupervisorRagInterface(BaseRagInterface):
    """Implementation of the RAG interface using VoxSigilRAG."""

    def __init__(
        self,
        voxsigil_library_path: Optional[Path] = None,
        embedding_model_name: Optional[str] = None,
    ):
        """Initialize the RAG interface with an optional VoxSigilRAG instance."""
        # Check if VoxSigilRAG is available
        if not VOXSIGIL_RAG_AVAILABLE:
            logger_rag_interface.warning(
                "VoxSigilRAG not available. Using mock implementation."
            )
            self.rag_instance = None
            return

        # Try to initialize VoxSigilRAG
        try:
            self.rag_instance = ImportedVoxSigilRAG(
                voxsigil_library_path=voxsigil_library_path
            )
            logger_rag_interface.info(
                "Initialized SupervisorRagInterface with VoxSigilRAG"
            )
        except Exception as e:
            logger_rag_interface.error(f"Failed to initialize VoxSigilRAG: {e}")
            self.rag_instance = None

    # Unified interface methods (implementing BaseRagInterface from Vanta)
    async def retrieve_documents(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query (unified interface method)."""
        # Map to existing sigil retrieval for backward compatibility
        return self.retrieve_sigils(query, top_k=k, filter_conditions=filters)
    
    async def index_document(
        self, 
        document: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Index a document for retrieval (unified interface method)."""
        # For now, return a placeholder - actual implementation would index the document
        logger_rag_interface.warning("Document indexing not implemented in SupervisorRagInterface")
        return f"doc_{len(str(document))}"
    
    async def augment_query(
        self, 
        query: str, 
        context: List[Dict[str, Any]]
    ) -> str:
        """Augment query with retrieved context (unified interface method)."""
        if not context:
            return query
        
        context_text = "\n".join([
            doc.get('content', doc.get('text', ''))[:200] + "..." 
            for doc in context[:3]
        ])
        
        return f"Context: {context_text}\n\nQuery: {query}"
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics (unified interface method)."""
        return {
            'rag_instance_available': self.rag_instance is not None,
            'interface_type': 'SupervisorRagInterface',
            'voxsigil_rag_available': VOXSIGIL_RAG_AVAILABLE
        }

    # Legacy methods (maintaining backward compatibility)
    def retrieve_sigils(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant sigils using VoxSigilRAG."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return []

        filter_conditions = filter_conditions or {}

        # Convert filter_conditions to VoxSigilRAG parameters
        rag_params = {
            "num_sigils": top_k,
            "detail_level": filter_conditions.get("detail_level", "standard"),
        }

        # Add tag filters if specified
        if "tags" in filter_conditions:
            rag_params["filter_tags"] = filter_conditions["tags"]
            rag_params["tag_operator"] = filter_conditions.get("tag_operator", "OR")

        # Add minimum similarity threshold if specified
        if "min_similarity" in filter_conditions:
            rag_params["min_score_threshold"] = filter_conditions["min_similarity"]

        try:
            # Call VoxSigilRAG to create context
            _, retrieved_sigils = self.rag_instance.create_rag_context(
                query=query, **rag_params
            )
            return retrieved_sigils if retrieved_sigils else []
        except Exception as e:
            logger_rag_interface.error(f"Error retrieving sigils: {e}")
            return []

    def retrieve_context(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Retrieves formatted context using VoxSigilRAG."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return "ERROR: VoxSigilRAG not available for context retrieval."

        params = params or {}

        # Extract parameters relevant to VoxSigilRAG
        scaffold = params.get("scaffold")

        # If we have a scaffold, adjust the context retrieval
        if scaffold:
            scaffold_tags = self._extract_scaffold_tags(scaffold)
            if scaffold_tags:
                params.setdefault("filter_tags", []).extend(scaffold_tags)

        # Convert params to VoxSigilRAG parameters
        rag_params = {
            "num_sigils": params.get("top_k", 5),
            "detail_level": params.get("detail_level", "standard"),
        }

        # Add tag filters if specified
        if "filter_tags" in params:
            rag_params["filter_tags"] = params["filter_tags"]
            rag_params["tag_operator"] = params.get("tag_operator", "OR")

        try:
            # Call VoxSigilRAG to create context
            context_text, _ = self.rag_instance.create_rag_context(
                query=query, **rag_params
            )
            return context_text if context_text else ""
        except Exception as e:
            logger_rag_interface.error(f"Error retrieving context: {e}")
            return f"ERROR: Failed to retrieve context: {e}"

    def retrieve_scaffolds(
        self, query: str, filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves reasoning scaffolds relevant to the query."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return []

        # Scaffolds are stored as sigils with the 'scaffold' tag
        scaffold_tags = ["scaffold"]
        if filter_tags:
            scaffold_tags.extend(filter_tags)

        try:
            # Use the existing sigil retrieval with scaffold tags
            return self.retrieve_sigils(
                query=query,
                top_k=5,
                filter_conditions={"tags": scaffold_tags, "detail_level": "full"},
            )
        except Exception as e:
            logger_rag_interface.error(f"Error retrieving scaffolds: {e}")
            return []

    def get_scaffold_definition(
        self, scaffold_name_or_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the full definition of a specific reasoning scaffold."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return None

        try:
            # First try to get by exact sigil ID
            sigil = self.get_sigil_by_id(scaffold_name_or_id)
            if sigil and "scaffold" in sigil.get("tags", []):
                return sigil

            # If not found by ID, search by name or description
            scaffolds = self.retrieve_scaffolds(
                query=scaffold_name_or_id, filter_tags=[scaffold_name_or_id.lower()]
            )

            # Return the first exact match or most relevant scaffold
            for scaffold in scaffolds:
                if (
                    scaffold.get("title", "").lower() == scaffold_name_or_id.lower()
                    or scaffold.get("sigil_glyph") == scaffold_name_or_id
                    or scaffold_name_or_id.lower() in scaffold.get("title", "").lower()
                ):
                    return scaffold

            return scaffolds[0] if scaffolds else None

        except Exception as e:
            logger_rag_interface.error(f"Error retrieving scaffold definition: {e}")
            return None

    def get_sigil_by_id(self, sigil_id_glyph: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific sigil by its unique ID or glyph."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return None

        try:
            # Use VoxSigilRAG to search for the specific sigil
            results = self.retrieve_sigils(
                query=sigil_id_glyph,
                top_k=1,
                filter_conditions={"detail_level": "full", "min_similarity": 0.95},
            )

            # Return exact match if found
            for result in results:
                if (
                    result.get("sigil_glyph") == sigil_id_glyph
                    or result.get("id") == sigil_id_glyph
                ):
                    return result

            return None

        except Exception as e:
            logger_rag_interface.error(f"Error retrieving sigil by ID: {e}")
            return None

    def _extract_scaffold_tags(self, scaffold_name: str) -> List[str]:
        """Extract relevant tags for a given scaffold to enhance context retrieval."""
        scaffold_tag_map = {
            "TRIALOGOS": ["problem_solving", "reasoning", "dialogue", "debate"],
            "C_STRUCTURE": ["algorithm", "code", "programming", "structure"],
            "SMART_MRAP": ["goal_oriented", "planning", "strategies"],
        }
        return scaffold_tag_map.get(scaffold_name, [])


class SimpleRagInterface(BaseRagInterface):
    """A simple implementation of the RAG interface that wraps a RAGProcessor."""

    def __init__(self, rag_processor):
        """Initialize with a RAGProcessor instance."""
        self.rag_processor = rag_processor
        logger_rag_interface.info("SimpleRagInterface initialized with RAGProcessor")

    # Unified interface methods (implementing BaseRagInterface from Vanta)
    async def retrieve_documents(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query (unified interface method)."""
        # Map to existing sigil retrieval for backward compatibility
        return self.retrieve_sigils(query, top_k=k, filter_conditions=filters)
    
    async def index_document(
        self, 
        document: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Index a document for retrieval (unified interface method)."""
        try:
            if hasattr(self.rag_processor, "index_document"):
                return self.rag_processor.index_document(document, metadata)
            elif hasattr(self.rag_processor, "add_document"):
                return self.rag_processor.add_document(document)
            else:
                logger_rag_interface.warning("Document indexing not supported by RAGProcessor")
                return f"simple_doc_{len(str(document))}"
        except Exception as e:
            logger_rag_interface.error(f"Error indexing document in SimpleRagInterface: {e}")
            return f"error_doc_{len(str(document))}"
    
    async def augment_query(
        self, 
        query: str, 
        context: List[Dict[str, Any]]
    ) -> str:
        """Augment query with retrieved context (unified interface method)."""
        if not context:
            return query
        
        context_text = "\n".join([
            doc.get('content', doc.get('text', ''))[:150] + "..." 
            for doc in context[:2]
        ])
        
        return f"Relevant context:\n{context_text}\n\nQuestion: {query}"
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics (unified interface method)."""
        stats = {
            'interface_type': 'SimpleRagInterface',
            'rag_processor_available': self.rag_processor is not None
        }
        
        try:
            if hasattr(self.rag_processor, "get_stats"):
                stats['processor_stats'] = self.rag_processor.get_stats()
            elif hasattr(self.rag_processor, "stats"):
                stats['processor_stats'] = self.rag_processor.stats
        except Exception as e:
            stats['stats_error'] = str(e)
        
        return stats

    # Legacy methods (maintaining backward compatibility)
    def retrieve_sigils(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant VoxSigil sigils by delegating to the RAGProcessor."""
        try:
            if hasattr(self.rag_processor, "retrieve"):
                return self.rag_processor.retrieve(
                    query, top_k=top_k, filters=filter_conditions
                )
            result = self.rag_processor.process(query)
            if isinstance(result, list):
                return result[:top_k]
            logger_rag_interface.warning(
                "RAGProcessor.process did not return a list of sigils"
            )
            return []
        except Exception as e:
            logger_rag_interface.error(
                f"Error in SimpleRagInterface.retrieve_sigils: {e}"
            )
            return []

    def retrieve_context(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Retrieves context as a string by processing the query through the RAGProcessor."""
        try:
            result = self.rag_processor.process(query)
            if isinstance(result, str):
                return result
            return str(result) if result else ""
        except Exception as e:
            logger_rag_interface.error(
                f"Error in SimpleRagInterface.retrieve_context: {e}"
            )
            return f"Error retrieving context: {e}"

    def retrieve_scaffolds(
        self, query: str, filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves reasoning scaffolds relevant to the query."""
        try:
            # Check if processor has a dedicated scaffold retrieval method
            if hasattr(self.rag_processor, "retrieve_scaffolds"):
                return self.rag_processor.retrieve_scaffolds(
                    query, filter_tags=filter_tags
                )

            # Fall back to general retrieval method with scaffold filter
            scaffold_tags = ["scaffold"]
            if filter_tags:
                scaffold_tags.extend(filter_tags)

            if hasattr(self.rag_processor, "retrieve"):
                return self.rag_processor.retrieve(
                    query, filters={"tags": scaffold_tags, "tag_operator": "AND"}
                )

            # Last resort: use generic process method and filter manually
            results = self.rag_processor.process(query)
            if not isinstance(results, list):
                logger_rag_interface.warning(
                    "RAGProcessor.process did not return a list for scaffold retrieval"
                )
                return []

            # Manual filtering for scaffold tags
            filtered_results = []
            for item in results:
                if isinstance(item, dict) and "tags" in item:
                    item_tags = item.get("tags", [])
                    if "scaffold" in item_tags and all(
                        tag in item_tags for tag in (filter_tags or [])
                    ):
                        filtered_results.append(item)

            return filtered_results

        except Exception as e:
            logger_rag_interface.error(
                f"Error in SimpleRagInterface.retrieve_scaffolds: {e}"
            )
            return []

    def get_scaffold_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific scaffold definition by name."""
        try:
            # Try dedicated scaffold retrieval method if available
            if hasattr(self.rag_processor, "get_scaffold"):
                return self.rag_processor.get_scaffold(name)

            # Try get_by_id if available, checking for scaffold tag
            if hasattr(self.rag_processor, "get_by_id"):
                scaffold = self.rag_processor.get_by_id(name)
                if scaffold and isinstance(scaffold, dict):
                    if "tags" in scaffold and "scaffold" in scaffold["tags"]:
                        return scaffold

            # Fall back to retrieve_scaffolds and find by name
            scaffolds = self.retrieve_scaffolds(name)
            for scaffold in scaffolds:
                if scaffold.get("name") == name or scaffold.get("id") == name:
                    return scaffold

            # If no exact match, use the most relevant scaffold if any were found
            if scaffolds:
                logger_rag_interface.info(
                    f"No exact scaffold match for '{name}', using most relevant result"
                )
                return scaffolds[0]

            logger_rag_interface.warning(f"No scaffold definition found for '{name}'")
            return None

        except Exception as e:
            logger_rag_interface.error(
                f"Error in SimpleRagInterface.get_scaffold_definition: {e}"
            )
            return None

    def get_sigil_by_id(self, sigil_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific sigil by ID."""
        try:
            # Try dedicated get_by_id method if available
            if hasattr(self.rag_processor, "get_by_id"):
                sigil = self.rag_processor.get_by_id(sigil_id)
                if sigil:
                    return sigil

            # Try search_by_id if available
            if hasattr(self.rag_processor, "search_by_id"):
                sigil = self.rag_processor.search_by_id(sigil_id)
                if sigil:
                    return sigil

            # Try retrieve with ID filter
            if hasattr(self.rag_processor, "retrieve"):
                results = self.rag_processor.retrieve(
                    sigil_id, filters={"id": sigil_id}
                )
                if results and len(results) > 0:
                    return results[0]

                # Also try with glyph field if ID didn't work
                results = self.rag_processor.retrieve(
                    sigil_id, filters={"glyph": sigil_id}
                )
                if results and len(results) > 0:
                    return results[0]

            # Last resort: use generic process method and filter manually
            results = self.rag_processor.process(sigil_id)
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        if item.get("id") == sigil_id or item.get("glyph") == sigil_id:
                            return item

                # If no exact match but we have results, return the first
                if results:
                    logger_rag_interface.info(
                        f"No exact sigil match for ID '{sigil_id}', using most relevant result"
                    )
                    return results[0]

            logger_rag_interface.warning(f"No sigil found with ID '{sigil_id}'")
            return None

        except Exception as e:
            logger_rag_interface.error(
                f"Error in SimpleRagInterface.get_sigil_by_id: {e}"
            )
            return None


# MockRagInterface removed - use Vanta.core.fallback_implementations.FallbackRagInterface instead
