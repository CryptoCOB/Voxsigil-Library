# voxsigil_supervisor/utils/validation_utils.py
"""
Utilities for validating data structures and inputs.
"""
from typing import Dict, Any, List, Tuple, Union

def validate_payload_structure(payload: Any, expected_keys: List[str], payload_name: str = "Payload") -> Tuple[bool, str]:
    """
    Validates if a dictionary payload contains all expected keys.
    """
    if not isinstance(payload, dict):
        return False, f"{payload_name} is not a dictionary."
    
    missing_keys = [key for key in expected_keys if key not in payload]
    if missing_keys:
        return False, f"{payload_name} is missing required keys: {', '.join(missing_keys)}."
    
    return True, f"{payload_name} structure is valid."

def check_llm_message_format(messages: Any) -> Tuple[bool, str]:
    """
    Validates if the messages list for an LLM call is correctly formatted.
    Each message should be a dict with "role" and "content" string keys.
    """
    if not isinstance(messages, list):
        return False, "Messages payload must be a list."
    if not messages: # Allow empty list for some edge cases, or make it an error
        return True, "Messages payload is an empty list." 

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message at index {i} is not a dictionary."
        if "role" not in msg or not isinstance(msg["role"], str):
            return False, f"Message at index {i} missing 'role' string."
        if "content" not in msg or not isinstance(msg["content"], str):
            return False, f"Message at index {i} missing 'content' string."
        if msg["role"] not in ["system", "user", "assistant", "tool"]: # Common roles
             return False, f"Message at index {i} has unrecognized role: {msg['role']}."
    return True, "Messages format is valid."