#!/usr/bin/env python
"""
Create a temporary module that exports the needed function
"""

from typing import Dict, List, Optional, Tuple, Any

# Import the internal implementation
try:
    from ARC.llm.arc_llm_handler import _llm_call_api_internal
except ImportError:
    # Direct import if the module is in the same directory
    try:
        from .arc_llm_handler import _llm_call_api_internal
    except ImportError:
        try:
            from arc_llm_handler import _llm_call_api_internal
        except ImportError:
            # Fallback implementation if the function is not available
            def _llm_call_api_internal(
                model_config: Dict[str, Any],
                messages_payload: List[Dict[str, str]],
                temperature: float,
                retry_attempt: int = 1,
            ) -> Tuple[Optional[str], Dict[str, Any]]:
                print(
                    "Warning: Using fallback _llm_call_api_internal function. This will not work correctly."
                )
                return "{'predicted_grid':[[0]]}", {}


# Create a public alias to the internal function
def call_llm_api(
    model_config: Dict[str, Any],
    messages_payload: List[Dict[str, str]],
    temperature: float,
    retry_attempt: int = 1,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Alias for the internal _llm_call_api_internal function to maintain compatibility."""
    return _llm_call_api_internal(
        model_config, messages_payload, temperature, retry_attempt
    )


# Export the public alias
__all__ = ["call_llm_api"]
