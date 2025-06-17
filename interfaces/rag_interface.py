# interfaces/rag_interface.py
"""
RAG Interface Implementations for VoxSigil Supervisor
====================================================

This module provides concrete implementations of the unified BaseRagInterface
from Vanta. All modules should now use the unified interface definition
from Vanta.interfaces instead of defining their own.

Legacy wrapper for backward compatibility - use Vanta.interfaces directly.
"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import unified interface from Vanta
from Vanta.interfaces.base_interfaces import BaseRagInterface

# Attempt to import your existing VoxSigilRAG
try:
    from VoxSigilRag import VoxSigilRAG  # Using the package import
    VOXSİGİL_RAG_AVAILABLE = True
    logger_rag_interface = logging.getLogger("VoxSigilSupervisor.interfaces.rag")
except ImportError:
    try:
        # Alternative: try direct module import
        from VoxSigilRag.voxsigil_rag import VoxSigilRAG
        VOXSİGİL_RAG_AVAILABLE = True 
        logger_rag_interface = logging.getLogger("VoxSigilSupervisor.interfaces.rag")
    except ImportError:
        VOXSİGİL_RAG_AVAILABLE = False
        # Create a dummy VoxSigilRAG if not found, so type hints don't break
        class VoxSigilRAG: 
            def __init__(self, *args, **kwargs): pass
            def inject_voxsigil_context(self, *args, **kwargs): return "", []
            def create_rag_context(self, *args, **kwargs): return "", []
        
        logger_rag_interface = logging.getLogger("VoxSigilSupervisor.interfaces.rag")
        logger_rag_interface.error(
            "Failed to import VoxSigilRAG. "
            "The SupervisorRagInterface will not be functional."
        )

# Note: BaseRagInterface is now imported from Vanta.interfaces
# This eliminates the duplicate interface definition


class SupervisorRagInterface(BaseRagInterface):
    """
    Implementation of the RAG interface using VoxSigilRAG.
    This wraps the existing VoxSigilRAG implementation to provide
    the interface needed by the Supervisor.
    """
    
    def __init__(self, voxsigil_library_path: Optional[Path] = None, embedding_model_name: Optional[str] = None):
        """
        Initialize the RAG interface with an optional VoxSigilRAG instance.
        
        Args:
            voxsigil_library_path: Optional path to the VoxSigil library.
            embedding_model_name: Optional name of the embedding model to use.
        """
        if not VOXSİGİL_RAG_AVAILABLE:
            logger_rag_interface.error("VoxSigilRAG not available. This interface will not function properly.")
            self.rag_instance = None
            return
            
        try:
            self.rag_instance = VoxSigilRAG(
                voxsigil_library_path=voxsigil_library_path,
                embedding_model_name=embedding_model_name
            )
            logger_rag_interface.info(f"Initialized SupervisorRagInterface with VoxSigilRAG")
        except Exception as e:
            logger_rag_interface.error(f"Failed to initialize VoxSigilRAG: {e}")
            self.rag_instance = None
    
    def retrieve_sigils(self, query: str, top_k: int = 5, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant sigils using VoxSigilRAG."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return []
        
        filter_conditions = filter_conditions or {}
        
        # Convert filter_conditions to VoxSigilRAG parameters
        rag_params = {
            'num_sigils': top_k,
            'detail_level': filter_conditions.get('detail_level', 'standard'),
        }
        
        # Add tag filters if specified
        if 'tags' in filter_conditions:
            rag_params['filter_tags'] = filter_conditions['tags']
            rag_params['tag_operator'] = filter_conditions.get('tag_operator', 'OR')
            
        # Add minimum similarity threshold if specified
        if 'min_similarity' in filter_conditions:
            rag_params['min_score_threshold'] = filter_conditions['min_similarity']
        
        try:
            # Call VoxSigilRAG to create context (but we only want the sigils, not the formatted text)
            _, retrieved_sigils = self.rag_instance.create_rag_context(query=query, **rag_params)
            return retrieved_sigils
        except Exception as e:
            logger_rag_interface.error(f"Error retrieving sigils: {e}")
            return []
    
    def retrieve_context(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Retrieves formatted context using VoxSigilRAG."""
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return "ERROR: VoxSigilRAG not available for context retrieval."
        
        params = params or {}
        
        # Extract parameters relevant to VoxSigilRAG
        scaffold = params.get('scaffold')
        
        # If we have a scaffold, adjust the context retrieval
        if scaffold:
            # Add scaffold-specific tags for better context
            scaffold_tags = self._extract_scaffold_tags(scaffold)
            if scaffold_tags:
                params.setdefault('filter_tags', []).extend(scaffold_tags)
        
        # Convert params to VoxSigilRAG parameters
        rag_params = {
            'num_sigils': params.get('top_k', 5),
            'detail_level': params.get('detail_level', 'standard'),
        }
        
        # Add tag filters if specified
        if 'filter_tags' in params:
            rag_params['filter_tags'] = params['filter_tags']
            rag_params['tag_operator'] = params.get('tag_operator', 'OR')
            
        try:
            # Call VoxSigilRAG to create context (we want the formatted text)
            context_text, _ = self.rag_instance.create_rag_context(query=query, **rag_params)
            return context_text
        except Exception as e:
            logger_rag_interface.error(f"Error retrieving context: {e}")
            return f"ERROR: Failed to retrieve context: {e}"
    
    def retrieve_scaffolds(self, query: str, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves reasoning scaffolds relevant to the query.
        This implementation leverages the VoxSigilRAG instance to find sigils that represent scaffolds.
        """
        if not self.rag_instance:
            logger_rag_interface.error("VoxSigilRAG instance not available")
            return []
            
        # Scaffolds are stored as sigils with the 'scaffold' tag
        scaffold_tags = ['scaffold']
        if filter_tags:
            scaffold_tags.extend(filter_tags)
            
        try:
            # Use the existing sigil retrieval with scaffold tags
            return self.retrieve_sigils(
                query=query,
                top_k=5,
                filter_conditions={
                    'tags': scaffold_tags,
                    'detail_level': 'full'  # We want the full scaffold definition
                }
            )
        except Exception as e:
            logger_rag_interface.error(f"Error retrieving scaffolds: {e}")
            return []
    
    def _extract_scaffold_tags(self, scaffold_name: str) -> List[str]:
        """Extract relevant tags for a given scaffold to enhance context retrieval."""
        # Map scaffold names to useful tags for context retrieval
        scaffold_tag_map = {
            "TRIALOGOS": ["problem_solving", "reasoning", "dialogue", "debate"],
            "C_STRUCTURE": ["algorithm", "code", "programming", "structure"],
            "SMART_MRAP": ["goal_oriented", "planning", "strategies"]
        }
        return scaffold_tag_map.get(scaffold_name, [])

    @abstractmethod
    def get_scaffold_definition(self, scaffold_name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full definition of a specific reasoning scaffold
        (which is also a VoxSigil construct, likely identified by a specific tag).

        Args:
            scaffold_name_or_id: The unique name or sigil glyph of the scaffold.

        Returns:
            A dictionary containing the scaffold's definition, or None if not found
            or not identified as a scaffold.
        """
        pass

    @abstractmethod
    def get_sigil_by_id(self, sigil_id_glyph: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific sigil by its unique ID or glyph.
        """
        pass


class SimpleRagInterface(BaseRagInterface):
    """
    A simple implementation of the RAG interface that wraps a RAGProcessor.
    This is a lightweight adapter used primarily for training integrations.
    """
    
    def __init__(self, rag_processor):
        """
        Initialize with a RAGProcessor instance.
        
        Args:
            rag_processor: An instance of RAGProcessor from VoxSigilRag
        """
        self.rag_processor = rag_processor
        logger_rag_interface.info("SimpleRagInterface initialized with RAGProcessor")
    
    def retrieve_sigils(self, query: str, top_k: int = 5, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant VoxSigil sigils by delegating to the RAGProcessor.
        
        Args:
            query: The query string to search for.
            top_k: The maximum number of sigils to return.
            filter_conditions: Optional dictionary for filtering.
            
        Returns:
            A list of dictionaries representing retrieved sigils.
        """
        try:
            # Delegate to the processor's retrieve method if available
            if hasattr(self.rag_processor, 'retrieve'):
                return self.rag_processor.retrieve(query, top_k=top_k, filters=filter_conditions)
            # Fallback to process method which might return context we can parse
            result = self.rag_processor.process(query)
            if isinstance(result, list):
                return result[:top_k]
            logger_rag_interface.warning("RAGProcessor.process did not return a list of sigils")
            return []
        except Exception as e:
            logger_rag_interface.error(f"Error in SimpleRagInterface.retrieve_sigils: {e}")
            return []
    
    def retrieve_context(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Retrieves context as a string by processing the query through the RAGProcessor.
        
        Args:
            query: The query string to search for.
            params: Optional parameters.
            
        Returns:
            A string containing the formatted context.
        """
        try:
            result = self.rag_processor.process(query)
            if isinstance(result, str):
                return result
            # If it returned something else, try to convert to string
            return str(result) if result else ""
        except Exception as e:
            logger_rag_interface.error(f"Error in SimpleRagInterface.retrieve_context: {e}")
            return f"Error retrieving context: {e}"
    
    def retrieve_scaffolds(self, query: str, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves reasoning scaffolds - in this simple implementation, 
        it returns empty list as scaffolds are not directly supported.
        
        Args:
            query: The query string to search for.
            filter_tags: Optional list of tags to filter scaffolds.
            
        Returns:
            An empty list as scaffolds are not directly supported.
        """
        logger_rag_interface.warning("SimpleRagInterface.retrieve_scaffolds called but not implemented")
        return []


if VOXSİGİL_RAG_AVAILABLE:
    class ActualVoxSigilRAGInterface(BaseRagInterface):
        """
        Concrete implementation of BaseRagInterface using the existing VoxSigilRAG module.
        """
        def __init__(self, voxsigil_rag_instance: VoxSigilRAG):
            self.rag_engine: VoxSigilRAG = voxsigil_rag_instance
            if not self.rag_engine._loaded_sigils: # Ensure sigils are loaded
                logger_rag_interface.info("ActualVoxSigilRAGInterface: Triggering load_all_sigils.")
                self.rag_engine.load_all_sigils()
            # Create an index for faster lookups if get_scaffold_definition needs it often
            self._sigil_index_by_id: Dict[str, Dict[str, Any]] = self.rag_engine.create_sigil_index()
            logger_rag_interface.info("ActualVoxSigilRAGInterface initialized and sigil index created.")

        def retrieve_sigils(self, query: str, top_k: int = 5, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
            logger_rag_interface.debug(f"Retrieving sigils for query: '{query[:50]}...' with top_k={top_k}, filters={filter_conditions}")
            
            rag_params: Dict[str, Any] = {
                "num_sigils": top_k,
                "query": query,
                "include_explanations": True, # Good for tracing, supervisor can decide to strip later
                # Set defaults from your VoxSigilRAG method signatures
                "filter_tag": None,
                "filter_tags": None,
                "tag_operator": "OR",
                "detail_level": "full", # Get all data for supervisor to use
                "min_score_threshold": 0.0, # Supervisor can filter later if needed
                "exclude_tags": None,
                "exclude_sigil_ids": None,
                "augment_query_flag": True,
                "apply_recency_boost_flag": True,
                "enable_context_optimizer": False, # Supervisor controls final context size differently
            }

            if filter_conditions:
                rag_params["filter_tags"] = filter_conditions.get("tags")
                rag_params["tag_operator"] = filter_conditions.get("tag_operator", "OR")
                rag_params["exclude_tags"] = filter_conditions.get("exclude_tags")
                rag_params["exclude_sigil_ids"] = filter_conditions.get("exclude_sigil_ids")
                rag_params["min_score_threshold"] = filter_conditions.get("min_score_threshold", 0.0)
            
            # The create_rag_context method returns: (formatted_sigil_context_as_string, list_of_retrieved_sigils_with_scores)
            # We are interested in the list_of_retrieved_sigils_with_scores for the interface.
            _formatted_context_str, retrieved_full_sigil_dicts = self.rag_engine.create_rag_context(**rag_params)
            
            logger_rag_interface.info(f"Retrieved {len(retrieved_full_sigil_dicts)} sigil items.")
            return retrieved_full_sigil_dicts

        def get_scaffold_definition(self, scaffold_name_or_id: str) -> Optional[Dict[str, Any]]:
            logger_rag_interface.debug(f"Getting scaffold definition for: {scaffold_name_or_id}")
            # Scaffolds are just sigils, possibly with a specific tag like "reasoning_scaffold"
            # We can use the index to find it by its sigil ID/glyph.
            scaffold_data = self._sigil_index_by_id.get(scaffold_name_or_id)
            
            if scaffold_data:
                # Optionally, verify if it's indeed a scaffold by checking its tags
                tags = scaffold_data.get("tags", [])
                if isinstance(tags, str): tags = [tags] # Ensure list
                if "reasoning_scaffold" in tags or "orchestration_pattern" in tags: # Example tags
                    logger_rag_interface.info(f"Found scaffold definition for '{scaffold_name_or_id}'.")
                    return scaffold_data
                else:
                    logger_rag_interface.warning(f"Construct '{scaffold_name_or_id}' found but not tagged as a scaffold.")
                    return None # Or return it anyway if strict typing isn't required
            
            logger_rag_interface.warning(f"Scaffold '{scaffold_name_or_id}' not found in sigil index.")
            return None

        def get_sigil_by_id(self, sigil_id_glyph: str) -> Optional[Dict[str, Any]]:
            logger_rag_interface.debug(f"Getting sigil by ID/glyph: {sigil_id_glyph}")
            return self._sigil_index_by_id.get(sigil_id_glyph)

else:
    raise ImportError(
        "VoxSigilRAG is not available. "
        "Ensure the VoxSigilRag package is installed and accessible."
    )