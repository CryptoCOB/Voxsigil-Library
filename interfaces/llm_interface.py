# voxsigil_supervisor/interfaces/llm_interface.py
"""
Defines the interface for Large Language Model (LLM) interaction
used by the VoxSigil Supervisor.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import sys
from pathlib import Path

# Setup the logger
logger_llm_interface = logging.getLogger("VoxSigilSupervisor.interfaces.llm")

# Attempt to import the LLM Handler
VOXSİGİL_LLM_HANDLER_AVAILABLE = False

# Define dummy functions for when the real ones aren't available
def dummy_llm_chat_completion(*args, **kwargs):
    return None, None, {"error": "LLM handler not available"}

def dummy_initialize_llm_handler(*args, **kwargs):
    return False

# The functions we'll use - start with dummies
llm_chat_completion = dummy_llm_chat_completion
initialize_llm_handler = dummy_initialize_llm_handler

try:
    # Try to import from ARC package
    from ARC.arc_llm_handler import llm_chat_completion as llm_chat_completion_import
    from ARC.arc_llm_handler import initialize_llm_handler as initialize_llm_handler_import
    llm_chat_completion = llm_chat_completion_import
    initialize_llm_handler = initialize_llm_handler_import
    VOXSİGİL_LLM_HANDLER_AVAILABLE = True
    logger_llm_interface.info("Successfully imported LLM handler from ARC package")
except ImportError:
    # Try direct module import
    try:
        # Add the ARC directory to the path
        arc_path = Path(__file__).resolve().parent.parent.parent / "ARC"
        if arc_path.exists():
            sys.path.append(str(arc_path))
            try:
                from ARC.arc_llm_handler import llm_chat_completion as llm_chat_completion_import
                from ARC.arc_llm_handler import initialize_llm_handler as initialize_llm_handler_import
                llm_chat_completion = llm_chat_completion_import
                initialize_llm_handler = initialize_llm_handler_import
                VOXSİGİL_LLM_HANDLER_AVAILABLE = True
                logger_llm_interface.info("Successfully imported LLM handler from ARC directory")
            except ImportError as e:
                logger_llm_interface.error(f"Failed to import from arc_llm_handler: {e}")
        else:
            logger_llm_interface.error(f"ARC path not found: {arc_path}")
    except Exception as e:
        logger_llm_interface.error(f"Failed to import LLM handler: {e}")


# Import unified interface from Vanta - replaces local definition
from Vanta.interfaces.base_interfaces import BaseLlmInterface

# Legacy comment: This file now imports the unified interface instead of defining its own


class SupervisorLlmInterface(BaseLlmInterface):
    """
    Implementation of the LLM interface using the existing LLM handler.
    """
    
    def __init__(self, force_initialize: bool = False, default_model_tier: int = 3):
        """
        Initialize the SupervisorLlmInterface.
        
        Args:
            force_initialize: Whether to force initialization of the LLM handler.
            default_model_tier: Default model strength tier to use (1-5).
        """
        self.logger = logger_llm_interface
        self.default_model_tier = default_model_tier
        self.default_temperature = 0.7
        self.available = VOXSİGİL_LLM_HANDLER_AVAILABLE
        
        if self.available:
            try:
                initialize_llm_handler(use_voxsigil_system_prompt=True, force_discover_models=force_initialize)
                self.logger.info("LLM handler initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM handler: {e}")
                self.available = False
        else:
            self.logger.error("LLM handler not available")
    
    def generate_response(self,
                         messages: List[Dict[str, str]],
                         task_requirements: Optional[Dict[str, Any]] = None,
                         temperature: Optional[float] = None,
                         system_prompt_override: Optional[str] = None,
                         use_global_system_prompt: bool = True
                         ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generates a response using the LLM handler.
        """
        if not self.available:
            self.logger.error("LLM handler not available for response generation")
            return None, None, {"error": "LLM handler not available"}
        
        task_requirements = task_requirements or {}
        temperature = temperature or self.default_temperature
        
        # Select model based on task requirements
        model_info = self.select_model(task_requirements)
        
        # Extract the final user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
        
        if not user_message:
            self.logger.error("No user message found in messages list")
            return None, None, {"error": "No user message found"}
        
        # Set up additional parameters based on task requirements
        llm_params = {
            "temperature": temperature,
            "system_prompt": system_prompt_override,
            "use_system_prompt": use_global_system_prompt,
            "service_priority": task_requirements.get("service_priority"),
            "model_strength_tier": task_requirements.get("model_tier", self.default_model_tier)
        }
        
        try:
            # Call the LLM handler
            response_text, model_used_info, response_metadata = llm_chat_completion(
                user_prompt=user_message,
                **llm_params
            )
            
            if response_text:
                self.logger.info(f"Generated response with model: {model_used_info.get('model_name', 'unknown')}")
            else:
                self.logger.error("Failed to generate response (empty response)")
                
            return response_text, model_used_info, response_metadata
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None, None, {"error": str(e)}
    
    def build_prompt(self, 
                    query: str, 
                    context: str, 
                    scaffold: Optional[str] = None, 
                    history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Builds a prompt with query, context, and optionally, a scaffold.
        """
        prompt_parts = []
        
        # Add history summary if available
        if history and len(history) > 0:
            prompt_parts.append("PREVIOUS ITERATIONS:")
            for i, h in enumerate(history[-2:]):  # Include up to 2 previous iterations
                prompt_parts.append(f"Attempt {i+1} Summary: {h.get('evaluation', {}).get('summary', 'No summary available')}")
            prompt_parts.append("")
        
        # Add scaffold if available
        if scaffold:
            prompt_parts.append(f"REASONING SCAFFOLD: {scaffold}")
            prompt_parts.append("")
        
        # Add context and query
        prompt_parts.append("CONTEXT:")
        prompt_parts.append(context)
        prompt_parts.append("\nQUERY:")
        prompt_parts.append(query)
        
        # Return the formatted prompt
        return "\n".join(prompt_parts)
    
    def select_model(self, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selects the most appropriate model based on task requirements.
        This is a simple implementation - your real implementation might be more complex.
        """
        # Default model info
        model_info = {
            "model_tier": task_requirements.get("model_tier", self.default_model_tier),
            "service_priority": task_requirements.get("service_priority", ["ollama", "lmstudio"])
        }
        
        # Adjust model tier based on task complexity
        if "complexity" in task_requirements:
            complexity = task_requirements["complexity"]
            if complexity == "high":
                model_info["model_tier"] = min(5, model_info["model_tier"] + 1)
            elif complexity == "low":
                model_info["model_tier"] = max(1, model_info["model_tier"] - 1)
        
        # Return the model info
        return model_info


class LocalLlmInterface(BaseLlmInterface):
    """
    A lightweight implementation of the LLM interface that works with local models
    for use in training integrations.
    """
    
    def __init__(self):
        """Initialize a simple local LLM interface."""
        self.logger = logger_llm_interface
        self.logger.info("LocalLlmInterface initialized")
    
    def generate_response(self,
                         messages: List[Dict[str, str]],
                         task_requirements: Optional[Dict[str, Any]] = None,
                         temperature: Optional[float] = None,
                         system_prompt_override: Optional[str] = None,
                         use_global_system_prompt: bool = True
                         ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generates a simple response for training purposes.
        
        In a real implementation, this would connect to a local LLM.
        For training purposes, this returns a simple placeholder response.
        """
        try:
            # Extract the user query
            user_message = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            # Create a simple response
            response_text = f"Response to: {user_message[:50]}..." if user_message else "No user query provided."
            model_info = {"model_name": "local_training_model", "provider": "local"}
            metadata = {"source": "training_integration", "is_placeholder": True}
            
            return response_text, model_info, metadata
        except Exception as e:
            self.logger.error(f"Error in LocalLlmInterface.generate_response: {e}")
            return f"Error: {e}", None, {"error": str(e)}
    
    def build_prompt(self, 
                    query: str, 
                    context: str, 
                    scaffold: Optional[str] = None, 
                    history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Builds a simple prompt for training purposes.
        """
        if context:
            return f"Context: {context}\n\nQuery: {query}"
        return query
    
    def select_model(self, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a fixed local model configuration for training.
        """
        return {
            "model_name": "local_training_model",
            "provider": "local",
            "temperature": 0.7,
            "model_tier": 1
        }